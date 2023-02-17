# GPTScore: Evaluate as You Desire

This is the Source Code of Paper: [GPTScore: Evaluate as You Desire](https://arxiv.org/pdf/2302.04166.pdf).


## What is GPTScore?

**GPTScore** is a novel evaluation framework that utilizes the emergent abilities (e.g., zero-shot instruction) of **G**enerative **P**re-**T**rained models to **Score** generated texts. 

<img src="./fig/framework.gif" width="800" class="center">

GPTScore evaluation framework support:

1. **Customizable**. Customized instructions and demonstrations enable the evaluation of new aspects without labeled datasets;
2. **Multifaceted**. One evaluator performs multifaceted evaluations;
3. **Training-free**.



## What PLMs does GPTScore support?

We explored 19 Pre-trained Language Models (PLMs) ranging in size from 80M (FLAN-T5-Small) to 175B (GPT3) to design GPTScore. <br>
The PLMs studied in this paper are listed as follows:

| **GPT3**         | **Param.** | **OPT**  | **Param.** | **FLAN-T5**      | **Param.** | **GPT2** | **Param.** |
|------------------|------------|----------|------------|------------------|------------|----------|------------|
| text-ada-001     | 350M       | OPT350M  | 350M       | FT5-small        | 80M        | GPT2-M   | 355M       |
| text-babbage-001 | 1.3B       | OPT-1.3B | 1.3B       | FT5-base         | 250M       | GPT2-L   | 774M       |
| text-curie-001   | 6.7B       | OPT-6.7B | 6.7B       | FT5-L            | 770M       | GPT2-XL  | 1.5B       |
| text-davinci-001 | 175B       | OPT-13B  | 13B        | FT5-XL           | 3B         | GPT-J-6B | 6B         |
| text-davinci-003 | 175B       | OPT-66B  | 66B        | FT5-XXL          | 11B        |          |            |



## Usage

Take the evaluation of `GPT3-text-curie-001` model as an example.

- Setting `gpt3_score` to `True`: the GPTScore evaluator uses a GPT3-based PLM.
- Setting `gpt3model` to `curie`: the  `text-curie-001` model is utilized.
- `out_dir_name`: set the folder for saving scoring results.
- `dataname`: set the dataset name for evaluation (e.g., `BAGEL`).
- `aspect`: set the aspect name to be evaluated (e.g., `quality`). 


### 1. GPTScore with Instruction and Demonstration
Set both the `use_demo` and `use_ist` as `True`. </br>
```
python score_d2t.py 
--device 'cuda:3' 
--dataname "BAGEL" 
--use_demo True 
--use_ist True 
--gpt3_score True 
--gpt3model "ada" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```


### 2. GPTScore with only Instruction
Set the `use_ist` to `True` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist True 
--gpt3_score True 
--gpt3model "ada" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```

### 3. GPTScore without both Instruction and Demonstration
Set the `use_ist` to `False` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist False 
--gpt3_score True 
--gpt3model "ada" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```





## Bib
```
@article{fu2023gptscore,
  title={GPTScore: Evaluate as You Desire},
  author={Fu, Jinlan and Ng, See-Kiong and Jiang, Zhengbao and Liu, Pengfei},
  journal={arXiv preprint arXiv:2302.04166},
  year={2023}
}
```