from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import os
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
from format_utils import transform_table_to_markdown

# Configuration
MAX_SEQ_LENGTH = 102400
DTYPE = None          # None for auto detection. 
LOAD_IN_4BIT = True   # Use 4-bit quantization to reduce memory usage.
EPOCHS = 9
TABLE_FORMAT = "markdown"
DATASET = "spider"
PROCESSED_DATA_DIR = f"dataset/{DATASET}/"
MODEL_PATH = "/home/scur2836/Tapex/IR2_table_reasoning/src/Table-Pretraining-main/models/Llama3.1_8B_Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
RESPONSE_TEMPLATE = "### Answer (be brief and exact):"
MAX_CHARACHTER_LENGTH = 250000 # Only applies for spider


def load_base_model():
    """
    Load the base language model and tokenizer using FastLanguageModel.
    """
    print(f"\n{'='*10} Loading base model {'='*10}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        local_files_only=True,
        trust_remote_code=True,
        model_type="llama"
    )
    return model, tokenizer


def apply_peft_to_model(model, tokenizer):
    """
    Apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
    """
    print(f"\n{'='*10} Applying PEFT model {'='*10}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def load_wikisql_dataset():
    """
    Load the WikiSQL dataset from the processed JSON files and filter entries.
    """
    print(f"\n{'='*10} Loading {DATASET} dataset {'='*10}")
    dataset = load_dataset(
        'json',
        data_files={
            'train': os.path.join(PROCESSED_DATA_DIR, 'train.json')
        }
    )
    
    if DATASET == "spider":
        # Filter entries with more than 15000 characters
        filtered_dataset = []
        discarded_count = 0

        for entry in dataset['train']:
            if len(entry['input']) + len(entry['output']) <= MAX_CHARACHTER_LENGTH:
                filtered_dataset.append(entry)
            else:
                discarded_count += 1  # Count discarded samples

        # Define the path for the filtered dataset
        filtered_data_path = os.path.join(PROCESSED_DATA_DIR, 'filtered_train.json')

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
            data_files={
                'train': filtered_data_path
            },
        )

    return dataset["train"]


def formatting_prompts_func(examples):
    """
    Convert raw input-output pairs into formatted prompt+answer text sequences.
    This function transforms the table into markdown (if specified), 
    then constructs a prompt that the model sees during training.
    """
    if not isinstance(examples["input"], list):
        input_texts = [examples["input"]]
        output_texts = [examples["output"]]
    else:
        input_texts = examples["input"]
        output_texts = examples["output"]
        
    formatted_texts = []

    for inp, out in zip(input_texts, output_texts):
        # Separate the question from the table
        question, _, table_part = inp.partition('col :')
        question = question.strip()
        table_str = 'col :' + table_part.strip()

        # Transform table to markdown if specified
        try:
            table = transform_table_to_markdown(table_str) if TABLE_FORMAT == "markdown" else table_str
        except Exception as e:
            print(f"Error processing example: {e}")
            print(f"Input text: {inp}")
            continue  # Skip example on error

        # Construct the prompt
        prompt = (
            "Below is a table and a question about its content. "
            "Provide a direct and concise answer to the question.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Table:\n{table}\n\n"
            f"{RESPONSE_TEMPLATE}"
        )

        # Append EOS token
        eos_token = tokenizer.eos_token if tokenizer.eos_token else "</s>"
        full_text = f"{prompt} {out}{eos_token}"
        formatted_texts.append(full_text)

    return formatted_texts


def create_data_collator(tokenizer):
    """
    Create the data collator that uses the response template to identify where the answer begins.
    """
    return DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False
    )


def create_sft_config():
    """
    Create a configuration object for the SFT trainer.
    """
    return SFTConfig(
        output_dir="outputs",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=0,
        learning_rate=2e-4,
        num_train_epochs=EPOCHS,
        # max_steps=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        packing=False,  # Must be false with DataCollatorForCompletionOnlyLM
    )


def train_model(model, tokenizer, train_dataset):
    """
    Train the model using the given dataset and configuration.
    """
    print(f"\n{'='*10} Training the model {'='*10}")
    collator = create_data_collator(tokenizer)
    sft_config = create_sft_config()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer_stats = trainer.train()
    return trainer_stats


def save_model_and_tokenizer(model, tokenizer):
    """
    Save the fine-tuned model and tokenizer to disk.
    """
    if DATASET == "spider":
        save_path = f"models/Llama_{DATASET}_epoch_{EPOCHS}_{TABLE_FORMAT}_{MAX_CHARACHTER_LENGTH}"
    else:
        save_path = f"models/Llama_{DATASET}_epoch_{EPOCHS}_{TABLE_FORMAT}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")


if __name__ == "__main__":
    # Load base model and tokenizer
    model, tokenizer = load_base_model()

    # Apply PEFT to the model
    model = apply_peft_to_model(model, tokenizer)

    # Load and prepare dataset
    train_dataset = load_wikisql_dataset()

    # Train the model
    train_stats = train_model(model, tokenizer, train_dataset)

    # Save the model and tokenizer
    save_model_and_tokenizer(model, tokenizer)

