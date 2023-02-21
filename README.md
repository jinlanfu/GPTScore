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


| **Model**        | **Parameter**   | **Evaluator Name**      | **Model**        | **Parameter**   | **Evaluator Name**      | 
|------------------|-----------------|-------------------------|------------------|-----------------|-------------------------|   
| **GPT3**         |                 |                         | **OPT**          |                 |                         |
| text-ada-001     | 350M            | gpt3\_score             | OPT350M          | 350M            | opt350m\_score          |
| text-babbage-001 | 1.3B            | gpt3\_score             | OPT-1.3B         | 1.3B            | opt1\_3B\_score         | 
| text-curie-001   | 6.7B            | gpt3\_score             | OPT-6.7B         | 6.7B            | opt6\_7B\_score         | 
| text-davinci-001 | 175B            | gpt3\_score             | OPT-13B          | 13B             | opt13B\_score           | 
| text-davinci-003 | 175B            | gpt3\_score             | OPT-66B          | 66B             | opt66B\_score           |  
| **FLAN-T5**      |                 |                         | **GPT2**         |                 |                         | 
| FT5-small        | 80M             | flan\_small\_score      | GPT2-M           | 355M            | gpt2\_medium\_score     | 
| FT5-base         | 250M            | flan\_base\_score       | GPT2-L           | 774M            | gpt2\_large\_score      |
| FT5-L            | 770M            | flan\_large\_score      | GPT2-XL          | 1.5B            | gpt2\_xl\_score         | 
| FT5-XL           | 3B              | flan\_xl\_score         | GPT-J-6B         | 6B              | gptJ6B\_score           |
| FT5-XXL          | 11B             | flan\_xxl\_score        |

* **Evaluator Name** indicates the name of the evaluator corresponding to the **Model** name in the first column.



## Usage


### Use the GPT3-based model as the evaluator

Take the evaluation of `GPT3-text-curie-001` model as an example.

- Setting `gpt3_score` to `True`: the GPTScore evaluator uses a GPT3-based PLM.
- Setting `gpt3model` to `curie`: the  `text-curie-001` model is utilized.
- `out_dir_name`: set the folder for saving scoring results.
- `dataname`: set the dataset name for evaluation (e.g., `BAGEL`).
- `aspect`: set the aspect name to be evaluated (e.g., `quality`). 


#### 1. GPTScore with Instruction and Demonstration
Set both the `use_demo` and `use_ist` as `True`. </br>
```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo True 
--use_ist True 
--gpt3_score True 
--gpt3model "curie" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```


#### 2. GPTScore with only Instruction
Set the `use_ist` to `True` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist True 
--gpt3_score True 
--gpt3model "curie" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```

#### 3. GPTScore without both Instruction and Demonstration
Set the `use_ist` to `False` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist False 
--gpt3_score True 
--gpt3model "curie" 
--out_dir_name "gpt3Score_based"  
--aspect 'quality'
```




### Use the non-GPT3-based model (e.g., OPT) as the evaluator
Here, we take the evaluation of `OPT350M` model as an example.

- Setting `opt350m_score` to `True`: use the evaluator named `opt350m_score`. 
- `out_dir_name`: set the folder for saving scoring results.
- `dataname`: set the dataset name for evaluation (e.g., `BAGEL`).
- `aspect`: set the aspect name to be evaluated (e.g., `quality`). 



#### 1. `opt350m_score` with Instruction and Demonstration
Set both the `use_demo` and `use_ist` as `True`. </br>
```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo True 
--use_ist True 
--opt350m_score True 
--out_dir_name "optScore_based"  
--aspect 'quality'
```


#### 2. `opt350m_score` with only Instruction
Set the `use_ist` to `True` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist True 
--opt350m_score True 
--out_dir_name "optScore_based"  
--aspect 'quality'
```


#### 3. `opt350m_score` without both Instruction and Demonstration
Set the `use_ist` to `False` and `use_demo` to `False`. </br>

```
python score_d2t.py 
--dataname "BAGEL" 
--use_demo False 
--use_ist False 
--opt350m_score True 
--out_dir_name "optScore_based"  
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