# IR2_table_reasoning

Reproduction of TAPEX pre-training and fine-tuning for table-based reasoning.

TAPEX pre-trains a model for table-based reasoning by simulating a neural SQL executor. Instead of relying on traditional NLP methods, TAPEX trains on SQL-like tasks over tables to improve structured data understanding, enabling it to handle question-answering and fact verification directly on tabular data. Evaluated on datasets like WikiSQL and TabFact, TAPEX demonstrates strong performance on table-centric tasks without needing SQL execution, using sequence generation to produce answers from table context alone.


## Reproduction Results

Denotation accuracies on WIKISQLWEAK.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | X | X |
| TAPEX (reproduced) | X | X |


Denotation accuracies on WIKITABLEQUESTIONS.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | X | X |
| TAPEX (reproduced) | X | X |

Denotation accuracies on SQA.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | X | X |
| TAPEX (reproduced) | X | X |

Accuracies on Tabfact.

| Model | Dev | Test |
|-------|-----|------|
| TAPEX | X | X |
| TAPEX (reproduced) | X | X |





