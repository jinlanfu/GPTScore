import time
import sys
from transformers import GPT2Tokenizer
import openai

class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger

    def do_inference(self, input, output, max_length=2048):
        losses = []
        data = input + output

        response = self.gpt3(data)
        out = response["choices"][0]

        assert input + output == out["text"]
        i = 0
        # find the end position of the input...
        i = out['logprobs']['text_offset'].index(len(input) - 1)
        if i == 0:
            i = i + 1

        print('eval text', out['logprobs']['tokens'][i: -1])
        loss = -sum(out['logprobs']["token_logprobs"][i:-1]) # ignore the last '.'
        avg_loss = loss / (len(out['logprobs']['text_offset']) - i-1) # 1 is the last '.'
        print('avg_loss: ', avg_loss)
        losses.append(avg_loss)

        return avg_loss


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=None):
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                print('prompt: ',prompt)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response

