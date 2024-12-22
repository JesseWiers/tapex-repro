# Introduction

The code for reproducing our paper "A Reproducibility Study of TAPEX". TAPEX pre-trains a model for table-based reasoning by simulating a neural SQL executor. Instead of relying on traditional NLP methods, TAPEX trains on SQL-like tasks over tables to improve structured data understanding, enabling it to handle question-answering and fact verification directly on tabular data. Evaluated on datasets like WikiSQL and TabFact, TAPEX demonstrates strong performance on table-centric tasks without needing SQL execution, using sequence generation to produce answers from table context alone.
We reproduce the original TAPEX PAPER and find that performance on WTQ and SQA can not be replicated. Moreover, we find that the performance on arithmetic operator is overstated and that TAPEX can be beaten with a stronger baseline.


## Quick Start

### Installation instructions

We use conda for package management, to create the environment run:
```bash
bash create_env.sh
```

Activate this environment to run the code without dependency problems:
```bash
conda activate tapex_repro
```

### Getting the pretrained checkpoints
It is possible to obtain the checkpoints of all models. [checkpoints/download_checkpoints.sh](checkpoints/download_checkpoints.sh) contains commands to download the various models. By default all commands are commented, uncomment the ones you are interested and run:

```bash
bash checkpoints/download_checkpoints.sh
```

### Running the code
The code and instructions to run all the main experiments can be found under [Table-Pretraining-main](Table-Pretraining-main). 

#### Evaluating on arithmetic operators 

To create the datasets for evaluating on different operators for WTQ run: 

```bash
python Table_Pretraining-main/examples/tableqa/process_wtq_data.py
```

Evaluating on these datasets is done in the same way as all the other datasets, detailed instructions can be found under [Table-Pretraining-main](Table-Pretraining-main) and [Table-Pretraining-main/examples/](Table-Pretraining-main/examples/)


#### Training Llama 

#### Training with data augmentations 

#### Training and evaluating on Spider




## Reproduction Results

Denotation accuracies on WIKISQLWEAK:

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 89.2 | 89.5 |
| TAPEX (reproduced) | 89.9 | 89.7 |


Denotation accuracies on WIKITABLEQUESTIONS:

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 57.0 | 57.5 |
| TAPEX (reproduced) | 53.5 | 52.0 |

Denotation accuracies on WIKITABLEQUESTIONS with different learning rates :

| Model | Learning rate 1e-5 | Learning rate 3e-5  | Learning rate 5e-5 | Learning rate 7e-5 |
|-------|-----|------|------|------|
| TAPEX | 0.488 | 0.519 | 0.502 | 0.531 |

Denotation accuracies on SQA:

| Model | Test |
|-------|------|
| TAPEX |  74.5 |
| TAPEX (reproduced) | 70.2 |

Accuracies on Tabfact:

| Model | Dev | Test | Test_simple | Test_complex | Test_small |
|-------|-----|------|------|------|------|
| TAPEX | 84.6 | 84.2 | 93.9 | 79.6 | 85.9 |  
| TAPEX (reproduced) | 83.5 | 83.44 | 93.93 | 78.41 | 85.14 |


## Additional Analysis

 Accuracies on different WTQ SQL operators:

| Model | avg | sum | count | max | min |
|-------|-----|------|------|------|------|
| TAPEX (Base) | 0.10 | 0.22 | 0.45 | 0.22 | 0.30 |
| TAPEX (Large) | 0.10 | 0.21 | 0.54 | 0.31 | 0.33 |  


Accuracies on spider-tableQA:

| Model | Dev |
|-------|------|
| TAPEX (Base) |0.172 |
| TAPEX (Large) | 0.280 |  

 Accuracies on different spider-tableQA SQL operators:

| Model | avg | sum | count | max | min |
|-------|-----|------|------|------|------|
| TAPEX (Base) | 0.00 | 0.065 | 0.202 | 0.10 | 0.037 |
| TAPEX (Large) |  0.048 | 0.032 | 0.288 | 0.325 | 0.259 |  
  