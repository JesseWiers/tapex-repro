
# Running the Llama 3.1 8B Instruct Model

## 1. Acquire the Llama Checkpoint

1. Download the **[Llama-3.1-8B-Instruct checkpoint](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** from Hugging Face.  
2. Move or copy the downloaded checkpoint folder into:
   ```
   Table-Pretraining-main/models/
   ```
   Make sure it includes both the tokenizer and model weights files.

---

## 2. Create and Activate the Environment

Create a new Conda environment (e.g., `unsloth_env`) with the required libraries and dependencies:

```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch \
    cudatoolkit \
    xformers \
    -c pytorch \
    -c nvidia \
    -c xformers \
    -y

conda activate unsloth_env
```

Then install the necessary Python packages:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
conda install -c conda-forge urllib3
conda install certifi
conda install -c conda-forge xxhashy
pip install matplotlib
```

---

## 3. Preprocess the Dataset

You can preprocess **WikiSQL**, **WTQ**, or **Spider** (or any custom dataset you have prepared in the same format) by running the corresponding scripts located in:

```
examples/tableqa/preprocess_{DATASET}_data.py
```

For example:

```bash
python examples/tableqa/preprocess_wikisql_data.py
```

> The preprocessed JSON files will be used by LLAMA, and the tokenized versions by TAPEX.

---

## 4. Verify `MODEL_PATH`

In the script:
```
llama/finetune_llama.py
```
make sure the variable **`MODEL_PATH`** points to the **directory** that holds the tokenizer and model weights (the folder you placed in `Table-Pretraining-main/models/`). In my example it is `Table-Pretraining-main/models/Llama3.1_8B_Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

---

## 5. Finetune the Model

Finetuning can be done for any of the supported datasets (`wikisql`, `wtq`, or `spider`). For example, to finetune for **3 epochs**:

```bash
conda activate unsloth_env
python llama/finetune_llama.py \
    --dataset wikisql \
    --epochs 3
```

> **Note**: The script automatically looks for the preprocessed JSON in the corresponding dataset folder.

---

## 6. Check the New Model

When finetuning completes, the new model is saved into:
```
models/
```
The final folder name depends on the dataset and the number of epochs. Verify the saved checkpoint.

---

## 7. Evaluate the Finetuned Model

To evaluate, use the matching dataset name and epoch count. For example:

```bash
conda activate unsloth_env
python llama/evaluate_llama.py \
    --dataset wikisql \
    --type_data test \
    --epochs 3
```

- **`--type_data`** selects the evaluation split or file, e.g., `test`, `valid`, or custom test sets (`avg_test`, `sum_test`, etc.), depending on how your data is named.
- The script will automatically pick the correct model checkpoint (based on dataset name and epochs) and evaluate on the chosen split.

---

## 8. Reading Results

The final lines of the evaluation script print out:
- **Denotation Accuracy**: an overall measure of how many predictions match the ground truth answers.
- A `.txt` file with prediction vs. ground-truth pairs is also saved to the model output directory.

> If you are testing on specialized test sets (e.g., `avg_test.json`, `sum_test.json`), pass `--type_data avg_test` (or the corresponding file name without `.json`) to evaluate on that specific set.

---