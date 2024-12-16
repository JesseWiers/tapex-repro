# IR2_tapex_reproducibility_study

Reproduction of TAPEX pre-training and fine-tuning for table-based reasoning.

TAPEX pre-trains a model for table-based reasoning by simulating a neural SQL executor. Instead of relying on traditional NLP methods, TAPEX trains on SQL-like tasks over tables to improve structured data understanding, enabling it to handle question-answering and fact verification directly on tabular data. Evaluated on datasets like WikiSQL and TabFact, TAPEX demonstrates strong performance on table-centric tasks without needing SQL execution, using sequence generation to produce answers from table context alone.


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

| Model | Learning rate 3e-4 | Learning rate 3e-5  | Learning rate 3e-6 | 
|-------|-----|------|------|
| TAPEX | 4.4 | 51.9 |  42.2|

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
| TAPEX (Base) | 0.00 | 0.065 | 0.202 | 0.10 |  |
| TAPEX (Large) |  0.048 | 0.032 | 0.288 | 0.325 | 0.259 |  
  



