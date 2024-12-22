from unsloth import FastLanguageModel
import torch
import os
from datasets import load_dataset
from tqdm import tqdm
import string
from format_utils import transform_table_to_markdown
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Evaluate the Llama model on a dataset.")
parser.add_argument('--dataset', type=str, default='wikisql', help='Name of the dataset to use (default: wikisql)')
parser.add_argument('--type_data', type=str, default='test', help='Type of data to evaluate (default: valid or test)')
parser.add_argument('--use_original_model', action='store_true', help='Use the original model instead of the fine-tuned model.')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')

args = parser.parse_args()

# Configuration
MAX_SEQ_LENGTH = 102400
MAX_CHARACHTER_LENGTH = 250000
DTYPE = None
LOAD_IN_4BIT = True
BATCH_SIZE = 1
TABLE_FORMAT = "markdown"
EPOCHS_USED = args.epochs
DATASET = args.dataset
MODEL_DIR = f"models/Llama_{args.dataset}_epoch_{EPOCHS_USED}_markdown"
PROCESSED_DATA_DIR = f"dataset/{args.dataset}/"
OUTPUT_FILE = f"llama/{args.type_data}_predictions_{args.dataset}_epoch_{EPOCHS_USED}_markdown.txt"
TYPE_DATA = args.type_data  # Use the provided type_data


def load_model_and_tokenizer(model_dir):
    """
    Load the fine-tuned model and tokenizer, then set the model in inference mode.
    """
    print(f"\n{'='*10} Loading model {'='*10}")
    
    # Check if the original model should be used
    if args.use_original_model:
        model_dir = "/home/scur2836/Tapex/IR2_table_reasoning/src/Table-Pretraining-main/models/Llama3.1_8B_Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"  # Specify the path to the original model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
        trust_remote_code=True,
        model_type="llama"
    )
    print(f"\n{'='*10} Loaded model {model_dir} {'='*10}")

    # Enable faster inference and set eval mode
    FastLanguageModel.for_inference(model)
    model.eval()

    return model, tokenizer


def load_test_dataset(data_dir):
    """
    Load the test split of the WikiSQL dataset.
    """
    print(f"\n{'='*10} Loading {TYPE_DATA} dataset {'='*10}")
    dataset =  load_dataset(
        'json',
        data_files={'test': os.path.join(data_dir, f'{TYPE_DATA}.json')}
    )

    if DATASET == "spider":
        # Filter entries with more than 15000 characters
        filtered_dataset = []
        discarded_count = 0

        for idx,entry in enumerate(dataset['test']):
            if len(entry['input']) + len(entry['output']) <= MAX_CHARACHTER_LENGTH:
                filtered_dataset.append(entry)
            else:
                discarded_count += 1  # Count discarded samples
                length = len(entry['input']) + len(entry['output'])
                print(f"{idx}: Example discarded with length: {length}\t ({round(length/MAX_CHARACHTER_LENGTH, 1)}x more than tolerated)")

        # Define the path for the filtered dataset
        filtered_data_path = os.path.join(PROCESSED_DATA_DIR, f'filtered_{TYPE_DATA}.json')

        # Save the filtered dataset to a new JSON file without square brackets
        with open(filtered_data_path, 'w') as f:
            import json
            for entry in filtered_dataset:
                json.dump(entry, f)
                f.write('\n')  # Write a newline after each JSON object

        # Print the number of discarded samples
        print(f"Number of discarded samples: {discarded_count}")
        print(f"Filtered dataset saved to: {filtered_data_path}")

        # Load the dataset using the newline-delimited format
        dataset = load_dataset(
            'json',
            data_files={'test': filtered_data_path},
        )
    return dataset['test']


def formatting_prompts_eval_func(examples):
    """
    Format a batch of input examples into inference prompts.
    The prompts contain the question and table but not the answer.
    """
    input_texts = examples["input"]
    formatted_texts = []

    for inp in input_texts:
        question, _, table_part = inp.partition('col :')
        question = question.strip()
        table_str = 'col :' + table_part.strip()

        try:
            table = transform_table_to_markdown(table_str) if TABLE_FORMAT == "markdown" else table_str
        except Exception as e:
            print(f"Error processing example: {e}\nInput text: {inp}")
            continue

        prompt = (
            "Below is a table and a question about its content. "
            "Provide a direct and concise answer to the question.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Table:\n{table}\n\n"
            "### Answer (be brief and exact):"
        )
        formatted_texts.append(prompt)

    return formatted_texts


def generate_answers_batch(model, tokenizer, batch_inputs):
    """
    Given a batch of raw inputs (question+table), generate answers from the model.
    """
    # Convert inputs to prompts
    prompts = formatting_prompts_eval_func({"input": batch_inputs})
    
    # Tokenize and move to device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Generate model predictions
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=1,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            early_stopping=True
        )
    
    # Decode answers and extract the part after the answer template
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = []
    for text in generated_texts:
        answer = text.split("Answer (be brief and exact):")[-1].strip()
        # Attempt numeric conversion
        try:
            if '.' in answer:
                answer = str(float(answer))
            else:
                answer = str(int(answer))
        except ValueError:
            # If it's not a number, just leave it as is
            pass
        answers.append(answer)
    
    return answers


def clean_text(text):
    """
    Clean the text by removing punctuation, except '.' and ',' 
    so that we don't break floats. Return a list of cleaned tokens.
    """
    translator = str.maketrans('', '', string.punctuation.replace('.', '').replace(',', ''))
    cleaned = text.translate(translator)
    return [word.strip(string.punctuation) for word in cleaned.split()]


def calculate_denotation_accuracy(predictions, truths):
    """
    Calculate denotation accuracy by comparing sets of cleaned tokens from predictions and ground truths.
    """
    correct = 0
    total = len(predictions)

    for pred, true in zip(predictions, truths):
        pred_set = set(clean_text(pred))
        true_set = set(clean_text(true))
        if pred_set == true_set:
            correct += 1

    return correct / total if total > 0 else 0


def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

    # Load test data
    dataset = load_test_dataset(PROCESSED_DATA_DIR)

    print(f"\n{'='*10} Starting evaluation {'='*10}")
    predictions = []
    ground_truth = []

    # Evaluate in batches
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    # for i in tqdm(range(0, 10 * BATCH_SIZE , BATCH_SIZE)):
        batch = dataset[i:i + BATCH_SIZE]
        batch_inputs = batch['input']
        batch_outputs = batch['output']

        batch_predictions = generate_answers_batch(model, tokenizer, batch_inputs)
        predictions.extend(batch_predictions)
        ground_truth.extend(batch_outputs)

        # Print sample results for the first batch
        # for j in range(min(5, len(batch_inputs))):
        #     print(f"\nInput: {batch_inputs[j][:100]}...")
        #     print(f"Predicted: {batch_predictions[j]}")
        #     print(f"True: {batch_outputs[j]}")
        #     print("-" * 50)

    # Compute accuracy
    accuracy = calculate_denotation_accuracy(predictions, ground_truth)
    correct_count = sum(1 for pred, true in zip(predictions, ground_truth) if pred == true)  # Count correct predictions
    total_count = len(ground_truth)  # Total number of examples
    print(f"\n{'='*10} Evaluation Results {'='*10}")
    print(f"Denotation Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")  # Updated print statement

    # Save predictions
    with open(OUTPUT_FILE, "w") as f:
        for pred, true in zip(predictions, ground_truth):
            f.write(f"Predicted: {pred}\n")
            f.write(f"True: {true}\n")
            f.write("-" * 50 + "\n")

    print(f"\nPredictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
