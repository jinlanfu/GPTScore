import argparse
import os
import time
import numpy as np
from utils import *
from gpt3_score import gpt3score
from transformers import GPT2Tokenizer
import json


class Scorer:
    """ Support GPT3-based (davinci, curie, babbage, ada), OPT-based, GPT2-based, FLAN-T5-based (19 models) """
    def __init__(self, args=None):
        self.args = args
        self.device = self.args.device
        self.eval_asp = self.args.aspect
        self.data = read_pickle(self.args.file_path)
        self.demos, self.asp_dfs = read_demos(self.args.demo_path)

        # Evaluate a small dataset first..
        print('Since GPT3-based models are expensive, we can test them on a small number of samples first.')
        print('The default number of test samples is 2.')
        import random
        random.seed(2)
        N = 2
        idxs = random.sample(range(0, len(self.data) - 1), N)
        new_data = {idx: self.data[idx] for idx in idxs}
        self.data = new_data
        print('the num of evaluation samples: ', len(self.data))

    def save_data(self, path):
        save_pickle(self.data, path)

    def demo_convert(self, demos, template):
        refhyp_demos = []
        hypref_demos = []
        for demo in demos:
            src_line = demo["src"].strip()
            ref_line = demo["ref_summ"].strip()
            hyp_line = demo["sys_summ"].strip()
            polar = demo["polarity"].strip()
            refhyp_demo = template.replace("XXXXX", ref_line).replace("YYYYY", hyp_line)
            refhyp_demos.append(refhyp_demo)
            hypref_demo = template.replace("XXXXX", hyp_line).replace("YYYYY",ref_line)
            hypref_demos.append(hypref_demo)
        return refhyp_demos, hypref_demos


    def score(self, metrics):
        """ metrics: list of metrics """

        for metric_name in metrics:
            if metric_name in [
                "opt125m_score", "opt350m_score", "opt1_3B_score",
                "opt2_7B_score", "opt6_7B_score", "opt13B_score", "opt30B_score", "opt66B_score",
                "gpt2_medium_score", "gpt2_large_score", "gpt2_xl_score", "gptJ6B_score"
                ]:
                """ Vanilla OPT and GPT2 models"""
                from opt_score import OPTScorer

                eval_asps = ["informativeness", "naturalness", "quality"]
                metric2checkpoint = {
                    "opt125m_score": "facebook/opt-125m",
                    "opt350m_score": "facebook/opt-350m",
                    "opt1_3B_score": "facebook/opt-1.3b",
                    "opt2_7B_score": "facebook/opt-2.7b",
                    "opt6_7B_score": "facebook/opt-6.7b",
                    "opt13B_score": "facebook/opt-13b",
                    "opt66B_score": "facebook/opt-66b",
                    "gpt2_medium_score": "gpt2-medium",
                    "gpt2_large_score": "gpt2-large",
                    "gpt2_xl_score": "gpt2-xl",
                    "gptJ6B_score": "EleutherAI/gpt-j-6B",
                }

                print('metric_name: ', metric_name)
                checkpoint = metric2checkpoint[metric_name]
                opt_scorer = OPTScorer(device=self.device, checkpoint=checkpoint)
                print(f'OPTScore setup finished. Begin calculating OPTScore.')

                start = time.time()
                for e_asp in eval_asps:
                    print('num of examples: ', len(self.data))
                    demo = self.demos[e_asp]
                    asp_df = self.asp_dfs[e_asp]
                    asp_df = asp_df.strip().replace(':', '. ')
                    print('demo: ', demo)
                    print('asp_df: ', asp_df)
                    refhyp_templates = ["XXXXX In other words , YYYYY", " In other words , "]
                    template = refhyp_templates[0]  # template
                    refhyp_demos, hypref_demos = self.demo_convert(demo, template)

                    for doc_id in self.data:
                        print('doc_id: ', doc_id)
                        ref_summs = self.data[doc_id]['ref_summs']
                        ref_summs = [add_dot(detokenize(line)) for line in ref_summs]  # avg, num_ref_summs:  127.65257142857143
                        sys_summ = add_dot(detokenize(self.data[doc_id]['sys_summ']))

                        ## ref->hypo
                        # define the prefix text...
                        if self.args.use_ist and self.args.use_demo:
                            refhyp_demos_str = "\n".join(refhyp_demos)
                            prefix = asp_df + '\n' + refhyp_demos_str + '\n'
                        elif self.args.use_ist and not self.args.use_demo:
                            prefix = asp_df + '\n'
                        elif not self.args.use_ist and not self.args.use_demo:
                            prefix = ''
                        # ref_summs1 = [prefix + template.replace('XXXXX', line).replace('YYYYY', '') for line in ref_summs]
                        ref_summs1 = [prefix + line for line in ref_summs]
                        ref_hypo_scores = np.array(opt_scorer.score(ref_summs1, [sys_summ] * len(ref_summs), prompt_text=refhyp_templates[1], batch_size=1))
                        ## hypo->ref
                        # define the prefix text...
                        if self.args.use_ist and self.args.use_demo:
                            hypref_demos_str = "\n".join(hypref_demos)
                            prefix = asp_df + '\n' + refhyp_demos_str + '\n'
                        elif self.args.use_ist and not self.args.use_demo:
                            prefix = asp_df + '\n'
                        elif not self.args.use_ist and not self.args.use_demo:
                            prefix = ''
                        sys_summ1 = [sys_summ] * len(ref_summs)
                        # sys_summ1 = [prefix + template.replace('XXXXX', line).replace('YYYYY', '') for line in sys_summ1]
                        sys_summ1 = [prefix + line for line in sys_summ1]
                        hypo_ref_scores = np.array(opt_scorer.score(sys_summ1, ref_summs, prompt_text=refhyp_templates[1],batch_size=1))

                        ref_hypo = ref_hypo_scores.max()
                        hypo_ref = hypo_ref_scores.max()
                        avg_f = (0.5 * (ref_hypo_scores + hypo_ref_scores)).max()
                        harm_f = (ref_hypo_scores * hypo_ref_scores / (ref_hypo_scores + hypo_ref_scores)).max()
                        print('ref_hypo: ', ref_hypo)
                        print('hypo_ref: ', hypo_ref)
                        print('avg_f: ', avg_f)
                        print('harm_f: ', harm_f)

                        if self.args.use_ist:
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_ref_hypo'] = ref_hypo
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_hypo_ref'] = hypo_ref
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_avg_f'] = avg_f
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_harm_f'] = harm_f
                        else:
                            self.data[doc_id]['scores'][f'{metric_name}_ref_hypo'] = ref_hypo
                            self.data[doc_id]['scores'][f'{metric_name}_hypo_ref'] = hypo_ref
                            self.data[doc_id]['scores'][f'{metric_name}_avg_f'] = avg_f
                            self.data[doc_id]['scores'][f'{metric_name}_harm_f'] = harm_f
                print(f'Finished calculating OPTScore, time passed {time.time() - start}s.')
                opt_scorer = None

            elif metric_name in ["flan_small_score", "flan_base_score", "flan_large_score","flan_xl_score", "flan_xxl_score"]:
                """ Vanilla flan_small, flan_base, flan_large, flan_xl, flan_xxl """
                from flan_score import FLANScorer

                eval_asps = ["informativeness", "naturalness", "quality"]
                metric2checkpoint = {
                    "flan_small_score": "google/flan-t5-small",
                    "flan_base_score": "google/flan-t5-base",
                    "flan_large_score": "google/flan-t5-large",
                    "flan_xl_score": "google/flan-t5-xl",
                    "flan_xxl_score": "google/flan-t5-xxl",
                }
                print('metric_name: ', metric_name)
                checkpoint = metric2checkpoint[metric_name]
                flan_scorer = FLANScorer(device=self.device, checkpoint=checkpoint)
                print(f'FLANScorer setup finished. Begin calculating FLANScorer.')

                start = time.time()
                for e_asp in eval_asps:
                    # Evaluation is cheap when using non-GPT3 models, so here, we evaluate all aspects by default.
                    print('num of examples: ', len(self.data))
                    demo = self.demos[e_asp]
                    asp_df = self.asp_dfs[e_asp]
                    asp_df = asp_df.strip().replace(':', '. ')
                    print('demo: ', demo)
                    print('asp_df: ', asp_df)
                    refhyp_templates = ["XXXXX In other words , YYYYY", ]
                    template = refhyp_templates[0]  # template
                    refhyp_demos, hypref_demos = self.demo_convert(demo, template)

                    for doc_id in self.data:
                        print('doc_id: ', doc_id)
                        ref_summs = self.data[doc_id]['ref_summs']
                        ref_summs = [add_dot(detokenize(line)) for line in ref_summs]  # avg, num_ref_summs:  127.65257142857143
                        sys_summ = add_dot(detokenize(self.data[doc_id]['sys_summ']))

                        ## ref->hypo
                        # define the prefix text...
                        if self.args.use_ist and self.args.use_demo:
                            refhyp_demos_str = "\n".join(refhyp_demos)
                            prefix = asp_df + '\n' + refhyp_demos_str + '\n'
                        elif self.args.use_ist and not self.args.use_demo:
                            prefix = asp_df + '\n'
                        elif not self.args.use_ist and not self.args.use_demo:
                            prefix = ''
                        ref_summs1 = [prefix + template.replace('XXXXX', line).replace('YYYYY', '')  for line in ref_summs]
                        ref_hypo_scores = np.array(flan_scorer.score(ref_summs1, [sys_summ] * len(ref_summs), batch_size=1))

                        ## hypo->ref
                        # define the prefix text...
                        if self.args.use_ist and self.args.use_demo:
                            hypref_demos_str = "\n".join(hypref_demos)
                            prefix = asp_df + '\n' + refhyp_demos_str + '\n'
                        elif self.args.use_ist and not self.args.use_demo:
                            prefix = asp_df + '\n'
                        elif not self.args.use_ist and not self.args.use_demo:
                            prefix = ''
                        sys_summ1 = [sys_summ] * len(ref_summs)
                        sys_summ1 = [prefix +  template.replace('XXXXX', line).replace('YYYYY', '') for line in sys_summ1]
                        hypo_ref_scores = np.array(flan_scorer.score(sys_summ1, ref_summs, batch_size=1))

                        ref_hypo = ref_hypo_scores.max()
                        hypo_ref = hypo_ref_scores.max()
                        avg_f = (0.5 * (ref_hypo_scores + hypo_ref_scores)).max()
                        harm_f = (ref_hypo_scores * hypo_ref_scores / (ref_hypo_scores + hypo_ref_scores)).max()
                        print('ref_hypo: ', ref_hypo)
                        print('hypo_ref: ', hypo_ref)
                        print('avg_f: ', avg_f)
                        print('harm_f: ', harm_f)

                        if self.args.use_ist:
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_ref_hypo'] = ref_hypo
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_hypo_ref'] = hypo_ref
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_avg_f'] = avg_f
                            self.data[doc_id]['scores'][f'{metric_name}_{e_asp}_harm_f'] = harm_f
                        else:
                            self.data[doc_id]['scores'][f'{metric_name}_ref_hypo'] = ref_hypo
                            self.data[doc_id]['scores'][f'{metric_name}_hypo_ref'] = hypo_ref
                            self.data[doc_id]['scores'][f'{metric_name}_avg_f'] = avg_f
                            self.data[doc_id]['scores'][f'{metric_name}_harm_f'] = harm_f
                print(f'Finished calculating FLANScorer, time passed {time.time() - start}s.')
                flan_scorer = None

            elif metric_name == 'gpt3_score':
                print(f'Perform the gpt3_score...')

                start = time.time()
                print('num of examples: ', len(self.data))
                demo = self.demos[self.eval_asp]
                asp_df = self.asp_dfs[self.eval_asp]
                print('demo: ', demo)
                print('asp_df: ', asp_df)
                refhyp_templates = ["XXXXX In other words , \nYYYYY",]
                template = refhyp_templates[0]  # template
                refhyp_demos, hypref_demos = self.demo_convert(demo, template)

                for samp_id, doc_id in enumerate(self.data):
                    print('samp_id: ', samp_id)
                    ref_summs = self.data[doc_id]['ref_summs']
                    ref_summs = [detokenize(line) for line in ref_summs]
                    sys_summ = detokenize(self.data[doc_id]['sys_summ'])

                    ref_hypo_scores = []
                    hypo_ref_scores = []

                    keep_seen_refsumm_score = {}
                    for k, ref_summ in enumerate(ref_summs):
                        print()
                        print('aspect: %s; samp_id: %d; ref_summ_id/total_ref_summ: %d/%d' % (
                        self.eval_asp, samp_id, k, len(ref_summs)))
                        # Add a period if missing punctuation at the end of the sentence.
                        ref_summ = add_dot(ref_summ)
                        sys_summ = add_dot(sys_summ)

                        if ref_summ in keep_seen_refsumm_score:
                            # skip the duplicate ref_summ
                            ref_hypo_score = keep_seen_refsumm_score[ref_summ][0]
                            hypo_ref_score = keep_seen_refsumm_score[ref_summ][1]
                            ref_hypo_scores.append(ref_hypo_score)
                            hypo_ref_scores.append(hypo_ref_score)
                        else:
                            ## ref->hypo
                            # define the prefix text...
                            if self.args.use_ist and self.args.use_demo:
                                refhyp_demos_str = "\n\n".join(refhyp_demos)
                                prefix = asp_df + '\n\n' + refhyp_demos_str + '\n\n'
                            elif self.args.use_ist and not self.args.use_demo:
                                prefix = asp_df + '\n'
                            elif not self.args.use_ist and not self.args.use_demo:
                                prefix = ''
                            input1 = template.replace("XXXXX", ref_summ).replace("YYYYY", "")
                            input1 = prefix + input1
                            output1 = lower_check(sys_summ)
                            ref_hypo_score = gpt3score(input1, output1, self.args.gpt3model, self.args.api_key)
                            ref_hypo_scores.append(ref_hypo_score)

                            ## hypo->ref
                            # define the prefix text...
                            if self.args.use_ist and self.args.use_demo:
                                hypref_demos_str = "\n\n".join(hypref_demos)
                                prefix = asp_df + '\n\n' + hypref_demos_str + '\n\n'
                            elif self.args.use_ist and not self.args.use_demo:
                                prefix = asp_df + '\n'
                            elif not self.args.use_ist and not self.args.use_demo:
                                prefix = ''
                            input2 = template.replace("XXXXX", sys_summ).replace("YYYYY", "")
                            input2 = prefix + input2
                            output2 = lower_check(ref_summ)
                            hypo_ref_score = gpt3score(input2, output2, self.args.gpt3model, self.args.api_key)
                            hypo_ref_scores.append(hypo_ref_score)

                            keep_seen_refsumm_score[ref_summ] =[ref_hypo_score,hypo_ref_score]
                    print('keep_seen_refsumm_score: ',keep_seen_refsumm_score)
                    print('len(ref_hypo_scores): ', len(ref_hypo_scores))
                    print('len(hypo_ref_scores): ', len(hypo_ref_scores))
                    ref_hypo_scores = np.array(ref_hypo_scores)
                    hypo_ref_scores = np.array(hypo_ref_scores)
                    ref_hypo = ref_hypo_scores.max()
                    hypo_ref = hypo_ref_scores.max()
                    avg_f = (0.5 * (ref_hypo_scores + hypo_ref_scores)).max()
                    harm_f = (ref_hypo_scores * hypo_ref_scores / (ref_hypo_scores + hypo_ref_scores)).max()
                    print('ref_hypo: ', ref_hypo)
                    print('hypo_ref: ', hypo_ref)
                    print('avg_f: ', avg_f)
                    print('harm_f: ', harm_f)

                    if self.args.use_ist:
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_ref_hypo'] = ref_hypo
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_hypo_ref'] = hypo_ref
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_avg_f'] = avg_f
                        self.data[doc_id]['scores'][f'{metric_name}_{self.eval_asp}_harm_f'] = harm_f
                    else:
                        self.data[doc_id]['scores'][f'{metric_name}_ref_hypo'] = ref_hypo
                        self.data[doc_id]['scores'][f'{metric_name}_hypo_ref'] = hypo_ref
                        self.data[doc_id]['scores'][f'{metric_name}_avg_f'] = avg_f
                        self.data[doc_id]['scores'][f'{metric_name}_harm_f'] = harm_f
                print(f'Finished calculating gpt3_score, time passed {time.time() - start}s.')

            else:
                raise NotImplementedError

def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file_path', type=str, default='XXX',
                        help='The data to load from.')
    parser.add_argument('--demo_path', type=str, default='XXX',
                        help='The demonstrated samples to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--dataname', type=str, default='BAGEL',
                        required=False, help='The name of the evaluated dataset.')
    parser.add_argument('--aspect', type=str, default='Engaging',
                        required=False, help='The evaluated aspect considered.')
    parser.add_argument('--gpt3model', type=str, default='ada',
                        required=False, help='Set which GPT3-based model to use.')
    parser.add_argument('--api_key', type=str, default='YOUR_OPENAI_API_KEY',
                        required=False, help='The OPENAI API key.')
    parser.add_argument('--use_ist', type=str2bool, default=False,
                        required=True, help='If set to True, use instruction.')
    parser.add_argument('--use_demo', type=str2bool, default=False,
                        required=False, help='If set to True, use demonstrated samples.')
    parser.add_argument('--output', type=str, default="XXXX",
                        help='The output path to save the calculated scores.')
    parser.add_argument('--out_dir_name', type=str, default="XXXX",
                        required=False, help='The output folder name to save the calculated scores.')

    parser.add_argument('--gpt3_score', type=str2bool,  default=False,
                        help='Whether to calculate gpt3_score.')
    parser.add_argument('--opt125m_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-125m.')
    parser.add_argument('--opt350m_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-350m.')
    parser.add_argument('--opt1_3B_score', type=str2bool, default=False,
                        help='Whether to calculate facebook/opt-1.3b.')
    parser.add_argument('--opt2_7B_score', type=str2bool, default=False,
                        help='Whether to calculate opt2_7B_score.')
    parser.add_argument('--opt6_7B_score', type=str2bool, default=False,
                        help='Whether to calculate opt6_7B_score.')
    parser.add_argument('--opt13B_score', type=str2bool, default=False,
                        help='Whether to calculate opt13B_score.')
    parser.add_argument('--opt66B_score', type=str2bool, default=False,
                        help='Whether to calculate opt66B_score.')
    parser.add_argument('--gpt2_medium_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_medium_score.')
    parser.add_argument('--gpt2_large_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_large_score.')
    parser.add_argument('--gpt2_xl_score', type=str2bool, default=False,
                        help='Whether to calculate gpt2_xl_score.')
    parser.add_argument('--gptJ6B_score', type=str2bool, default=False,
                        help='Whether to calculate gptJ6B_score.')
    parser.add_argument('--flan_small_score', type=str2bool, default=False,
                        help='Whether to calculate flan_small_score.')
    parser.add_argument('--flan_base_score', type=str2bool, default=False,
                        help='Whether to calculate flan_base_score.')
    parser.add_argument('--flan_large_score', type=str2bool, default=False,
                        help='Whether to calculate flan_large_score.')
    parser.add_argument('--flan_xl_score', type=str2bool, default=False,
                        help='Whether to calculate flan_xl_score.')
    parser.add_argument('--flan_xxl_score', type=str2bool, default=False,
                        help='Whether to calculate flan_xxl_score.')
    args = parser.parse_args()


    METRICS = []
    if args.gpt3_score:
        METRICS.append('gpt3_score')
    if args.opt350m_score:
        METRICS.append('opt350m_score')
    if args.opt1_3B_score:
        METRICS.append('opt1_3B_score')
    if args.opt6_7B_score:
        METRICS.append('opt6_7B_score')
    if args.opt13B_score:
        METRICS.append('opt13B_score')
    if args.opt66B_score:
        METRICS.append('opt66B_score')

    if args.flan_small_score:
        METRICS.append('flan_small_score')
    if args.flan_base_score:
        METRICS.append('flan_base_score')
    if args.flan_large_score:
        METRICS.append('flan_large_score')
    if args.flan_xl_score:
        METRICS.append('flan_xl_score')
    if args.flan_xxl_score:
        METRICS.append('flan_xxl_score')

    if args.gpt2_medium_score:
        METRICS.append('gpt2_medium_score')
    if args.gpt2_large_score:
        METRICS.append('gpt2_large_score')
    if args.gpt2_xl_score:
        METRICS.append('gpt2_xl_score')
    if args.gptJ6B_score:
        METRICS.append('gptJ6B_score')

    print('METRICS: ',METRICS)

    out_dir1 = args.out_dir_name

    data_dir = "./datas/meta_datas/d2t/"
    demon_dir = "./datas/demos/d2t/"
    out_dir = './analysis/d2t/'+out_dir1
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_name = args.dataname
    print('##### eval data_name: ', data_name)
    print()
    file_path = data_dir + data_name+'/data.pkl'
    args.file_path = file_path
    args.demo_path =demon_dir + data_name + '_demos.json'

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # args.output = out_dir + '/' + data_name + '_15plainBaseline_score.pkl'
    args.output = out_dir + '/' + data_name + args.out_dir_name + '_usedemo[' + str(args.use_demo) + ']' + '_useist[' + str(args.use_ist) + ']_OptFlanGpt2_score.pkl'
    # args.output = out_dir + '/' + data_name + '_' + args.aspect + '_' + args.out_dir_name + '_usedemo[' + str(args.use_demo) + ']' + '_useist[' + str(args.use_ist) + ']_' + args.gpt3model + '.pkl'
    print('args.output: ', args.output)

    scorer = Scorer(args)
    scorer.score(METRICS)
    scorer.save_data(args.output)

if __name__ == '__main__':
    main()

