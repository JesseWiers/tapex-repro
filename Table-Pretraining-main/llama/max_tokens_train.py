from unsloth import FastLanguageModel
import os
from datasets import load_dataset
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.pyplot as plt
import time
import signal
import threading
import numpy as np

# Configuration
DATASET = "wikisql"  # Change this to your dataset if needed
PROCESSED_DATA_DIR = f"dataset/{DATASET}/"
MODEL_PATH = "/home/scur2836/Tapex/IR2_table_reasoning/src/Table-Pretraining-main/models/Llama3.1_8B_Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

def load_model_and_tokenizer():
    """
    Load the base language model and tokenizer using FastLanguageModel.
    """
    print(f"\n{'='*10} Loading base model {'='*10}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        model_type="llama"
    )
    return model, tokenizer

def load_wikisql_dataset():
    """
    Load the WikiSQL dataset from the processed JSON files.
    """
    print(f"\n{'='*10} Loading {DATASET} dataset {'='*10}")
    dataset = load_dataset(
        'json',
        data_files={
            'train': os.path.join(PROCESSED_DATA_DIR, 'train.json')
        }
    )
    return dataset['train']

def tokenize_with_timeout(tokenizer, input_text, output_text, timeout):
    """
    Function to tokenize input and output with a timeout.
    """
    def target():
        nonlocal tokens
        tokens = tokenizer(input_text + " " + output_text, return_tensors="pt")

    tokens = None
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)  # Wait for the thread to finish or timeout

    if thread.is_alive():
        print("Tokenization took too long, skipping this example.")
        return None  # Indicate that tokenization failed
    else:
        return tokens

def plot_and_save_distribution(token_lengths, dataset_name, step):
    """
    Plot the distribution of token lengths and save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='blue', alpha=0.7)
    plt.title(f'Distribution of Token Lengths (Step {step})')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Ensure the directory exists before saving the figure
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Create directory if it doesn't exist

    plt.savefig(os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}_token_length_distribution_step_{step}.png'))
    plt.close()  # Close the figure to free memory

def calculate_max_tokens(train_dataset, tokenizer):
    """
    Calculate the maximum number of tokens for input + output in the training dataset
    and save the distribution of token lengths as a histogram.
    """
    max_tokens = 0
    total_tokens_count = 0  # Total number of tokens processed
    num_examples = 0  # Count of examples processed
    token_lengths = []  # List to store token lengths
    predefined_token_length = 1000000  # Example token length to use for discarded examples

    # Use tqdm to show progress
    with tqdm(total=len(train_dataset), desc="Calculating max tokens", unit="example") as pbar:
        for i, example in enumerate(train_dataset):
            input_text = example["input"]
            output_text = example["output"]

            # Tokenize input and output with a timeout of 0.1 seconds
            tokens = tokenize_with_timeout(tokenizer, input_text, output_text, timeout=2)

            if tokens is None:
                # Use predefined token length if tokenization failed
                continue
            else:
                total_tokens = tokens["input_ids"].size(1)  # Get the number of tokens

            # Store the token length
            token_lengths.append(total_tokens)  # Directly append total_tokens

            # Update max_tokens if the current total is greater
            if total_tokens > max_tokens:
                max_tokens = total_tokens
            
            # Update total tokens count and number of examples
            total_tokens_count += total_tokens
            num_examples += 1

            # Calculate average tokens
            average_tokens = total_tokens_count / num_examples if num_examples > 0 else 0
            
            # Calculate median
            median_tokens = np.median(token_lengths)

            # Identify outliers using IQR
            q1 = np.percentile(token_lengths, 25)
            q3 = np.percentile(token_lengths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [x for x in token_lengths if x < lower_bound or x > upper_bound]

            # Update the tqdm description with the current max_tokens, average_tokens, and median_tokens
            pbar.set_postfix(max_tokens=max_tokens, average_tokens=average_tokens, median_tokens=median_tokens)
            pbar.update(1)  # Increment the progress bar

    # Final plot after processing all examples
    plot_and_save_distribution(token_lengths, DATASET, "final")

    return max_tokens

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Load the training dataset
    train_dataset = load_wikisql_dataset()

    # Calculate the maximum tokens
    max_tokens = calculate_max_tokens(train_dataset, tokenizer)

    print(f"Maximum tokens per data point (input + output): {max_tokens}")
