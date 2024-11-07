# IR2_table_reasoning

Reproduction of TAPEX pre-training and fine-tuning for table-based reasoning.

TAPEX pre-trains a model for table-based reasoning by simulating a neural SQL executor. Instead of relying on traditional NLP methods, TAPEX trains on SQL-like tasks over tables to improve structured data understanding, enabling it to handle question-answering and fact verification directly on tabular data. Evaluated on datasets like WikiSQL and TabFact, TAPEX demonstrates strong performance on table-centric tasks without needing SQL execution, using sequence generation to produce answers from table context alone.


## Reproduction Results

Denotation accuracies on WIKISQLWEAK.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 89.2 | 89.5 |
| TAPEX (reproduced) | 89.9 | 89.6 |


Denotation accuracies on WIKITABLEQUESTIONS.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | 57.0 | 57.5 |
| TAPEX (reproduced) | 49.3 | 48.7 |

Denotation accuracies on SQA .

| Model | Test |
|-------|------|
| TAPEX |  74.5 |
| TAPEX (reproduced) | 0.613 |

Accuracies on Tabfact.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | X | X |
| TAPEX (reproduced) | X | X |





