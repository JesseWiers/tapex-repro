# IR2_tapex_reproducibility_study

Reproduction of TAPEX pre-training and fine-tuning for table-based reasoning.

TAPEX pre-trains a model for table-based reasoning by simulating a neural SQL executor. Instead of relying on traditional NLP methods, TAPEX trains on SQL-like tasks over tables to improve structured data understanding, enabling it to handle question-answering and fact verification directly on tabular data. Evaluated on datasets like WikiSQL and TabFact, TAPEX demonstrates strong performance on table-centric tasks without needing SQL execution, using sequence generation to produce answers from table context alone.


## Reproduction Results

Denotation accuracies on WIKISQLWEAK:

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 89.2 | 89.5 |
| TAPEX (reproduced) | 89.9 (±0.3)  | 89.7 (±0.1)  |


Denotation accuracies on WIKITABLEQUESTIONS:

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 57.0 | 57.5 |
| TAPEX (reproduced) | 53.5 (±3.0) | 52.0 (±2.3) |

Denotation accuracies on SQA:

| Model | Test |
|-------|------|
| TAPEX |  74.5 |
| TAPEX (reproduced) | 70.2 (±1.7) |

Accuracies on Tabfact:

| Model | Dev | Test | Test_simple | Test_complex | Test_small |
|-------|-----|------|------|------|------|
| TAPEX | 84.6 | 84.2 | 93.9 | 79.6 | 85.9 |  
| TAPEX (reproduced) | 83.5 (±0.1) | 83.44 (±0.1) | 93.93 (±0.13) | 78.41 (±0.15) | 85.14 (±0.28) |





