# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import zipfile
import logging
import pandas as pd
from datasets import load_dataset

from tqdm import tqdm

from tapex.common.download import download_file
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.format_converter import convert_fairseq_to_hf

RAW_DATASET_FOLDER = "raw_dataset"
PROCESSED_DATASET_FOLDER = "dataset"
TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
# Options: bart.base, bart.large, tapex.base, tapex.large
MODEL_NAME = "tapex.base"
logger = logging.getLogger(__name__)


def build_spider_fairseq_dataset(out_prefix, data_dir, operator=""):
    """
    Builds a Fairseq dataset for Spider

    Parameters:
    - out_prefix (str): The prefix for the output files. 'train' or 'valid'.
    - data_dir (str): The directory where the processed dataset will be saved.

    Returns:
    None
    """
    
    # Create output dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    input_f = open("{}/{}.src".format(data_dir, out_prefix), "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(data_dir, out_prefix), "w", encoding="utf8")

    spider_tableQA = load_dataset("vaishali/spider-tableQA")
    if out_prefix == "valid":
        df = pd.DataFrame(spider_tableQA["validation"])
        
        if operator != "":
            print(f"operator: {operator}")
            df = df[df['query'].str.contains(r'\b' + operator + r'\b', case=False, na=False)]
            print(f"DataFrame for operator '{operator}': {len(df)} rows")
            print(df['query'])
    else: 
        df = pd.DataFrame(spider_tableQA[out_prefix])
        

    for row, sample in tqdm(df.iterrows(), total=df.shape[0]):
        
        question = sample['question'].lower()
        input_tables = [pd.read_json(table, orient='split') for table in sample['tables']]
        answer = pd.read_json(sample['answer'], orient='split')
        
        if input_tables:  
            table_df = input_tables[0]  
            table_content = " ".join(
                [f"row {i + 1} : " + " | ".join(map(str, row)).replace('\n', ' ') for i, row in table_df.iterrows()]
            )
            input_source = f"{question} col : " + " | ".join(table_df.columns) + " " + table_content
        else:
            continue  

        if answer.shape[0] > 0: 
            output_target = " | ".join(
                [" | ".join(map(str, row)) for row in answer.values]
            ).replace('\n', ' ')
        else:
            output_target = ""  
            continue 
        
        input_source = input_source.lower()
        output_target = output_target.replace(" |", ",").lower()

        if input_source and output_target:  
            input_f.write(input_source + "\n")
            output_f.write(output_target + "\n")
               
    input_f.close()
    output_f.close()


def build_spider_huggingface_dataset(fairseq_data_dir):
    convert_fairseq_to_hf(fairseq_data_dir, "train")
    convert_fairseq_to_hf(fairseq_data_dir, "valid")


def preprocess_spider_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME, with_test_set=False)
    fairseq_binary_translation(processed_data_dir, resource_name=MODEL_NAME, with_test_set=False)


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))
    
    operators = ['max', 'min', 'avg', 'sum', 'count']
    
    for operator in operators:

        processed_spider_data_dir = os.path.join(PROCESSED_DATASET_FOLDER, f"spider_{operator}")

        logger.info("*" * 80)
        logger.info("Process the dataset and save the processed dataset in {}".format(processed_spider_data_dir))
        
        logger.info("Building the validation dataset in {}".format(processed_spider_data_dir))
        build_spider_fairseq_dataset("valid", processed_spider_data_dir, operator=operator)
        
        logger.info("Building the train dataset in {}".format(processed_spider_data_dir))
        build_spider_fairseq_dataset("train", processed_spider_data_dir)
        
        logger.info("*" * 80)
        logger.info("Begin to BPE and build the dataset binaries in {}/bin".format(processed_spider_data_dir))
        preprocess_spider_dataset(processed_spider_data_dir)

        logger.info("*" * 80)
        logger.info("Begin to build the HuggingFace dataset version in {}".format(processed_spider_data_dir))
        build_spider_huggingface_dataset(processed_spider_data_dir)

        logger.info("*" * 80)
        logger.info("Now you can train models using {} as the <data_dir> argument. "
                    "More details in `run_model.py`.".format(os.path.join(processed_spider_data_dir, MODEL_NAME)))