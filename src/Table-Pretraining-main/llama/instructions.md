# Running the Llama models
1. First install the Llama checkpoint model of 3.1 8B Intruct (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and move it into *Table-Pretraining-main/models* 
2. Install the environment as follows:
```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env
source activate unsloth_env
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
conda install -c conda-forge urllib3
conda install certifi
conda install -c conda-forge xxhashy
pip install matplotlib
```
3. Run the scripts in *examples/tableqa/preprocess_{DATASET}_data.py* to get the data (used for both TAPEX and LLAMA)
4. Make sure that in *finetune_llama.py* MODEL_PATH points to the snapshot that holds the tokenizer and model weights that you downloaded.
5. Finetune as follows:
```
conda activate unsloth_env
python llama/finetune_llama.py --dataset choose:[wikisql, wtq, spider] --epochs 3
```
6. The new model is saved in *models/*. Check if correct. The name is dependent on dataset and epochs
7. Evaluate as follows (it automatically picks right model, given correct dataset and epochs used to train):
```
conda activate unsloth_env
python llama/evaluate_llama.py --dataset choose:[wikisql, wtq, spider] --type_data choose:[test, valid] --epochs 3
```
8. End of script prints denotation accuracy and puts all prediction vs ground truth in a .txt file.

Note that *type_data* refers to the file name of the preprocessed .json inside the directory of the dataset. We made different test sets for the operators to test performance on this. So in those cases *type_data* is equal to avg_test, sum_test ... etc.
