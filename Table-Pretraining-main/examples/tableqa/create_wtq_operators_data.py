import pandas as pd
import json
import matplotlib.pyplot as plt
import subprocess
import os


def main():
    
    subprocess.run(['wget', '--header=Referer: https://huggingface.co/', '-P', 'raw_dataset/squall/', 'https://huggingface.co/JesseWiers/tapex_repro/resolve/main/squall.json'])
    
    os.makedirs('raw_dataset/squall/', exist_ok=True)

    with open('raw_dataset/squall/squall.json', 'r') as f:
        data = json.load(f)

    # Convert JSON to DataFrame
    df = pd.DataFrame(data)

    def flatten_sql(sql_tokens):
        return " ".join([token[1] for token in sql_tokens])

    # Adds new 'flattened_sql' column to the dataset
    df['flattened_sql'] = df['sql'].apply(flatten_sql)
    
    def flatten_nl(nl_tokens):
        return " ".join(nl_tokens)

    # Apply the function to the 'nl' column
    df['flattened_nl'] = df['nl'].apply(flatten_nl)
    
    # Define SQL operators
    sql_operators = {'sum', 'max', 'avg', 'min', 'count'}

    # Initialize a dictionary to store DataFrames for each operator
    operator_dfs = {}

    # Iterate through each operator and create a DataFrame for each
    for operator in sql_operators:
        operator_df = df[df['flattened_sql'].str.contains(r'\b' + operator + r'\b', case=False, na=False)]
        operator_dfs[operator] = operator_df

    # Print the number of rows for each operator DataFrame
    for operator, operator_df in operator_dfs.items():
        print(f"DataFrame for operator '{operator}': {len(operator_df)} rows")
        
    # Load validation, test, and train DataFrames
    validation_file_path = 'raw_dataset/wtq/data/random-split-1-dev.tsv'  
    val_df = pd.read_csv(validation_file_path, sep='\t', on_bad_lines='skip')
    print(f"Validation DataFrame rows: {val_df.shape[0]}")

    test_file_path = 'raw_dataset/wtq/data/pristine-unseen-tables.tsv' 
    test_df = pd.read_csv(test_file_path, sep='\t')
    print(f"Test DataFrame rows: {test_df.shape[0]}")

    train_file_path = 'raw_dataset/wtq/data/random-split-1-train.tsv'  
    train_df = pd.read_csv(train_file_path, sep='\t')
    print(f"Train DataFrame rows: {train_df.shape[0]}")
    
    # Clean 'tbl' column in DataFrames
    for df in [test_df, val_df, train_df]:
        df['tbl'] = df['context'].str.replace('csv/', '').str.replace('.csv', '').str.replace('-', '_', regex=False)

    # Adding 'nt' column to data partitions
    for df in [test_df, val_df, train_df]:
        df['nt'] = df['id']

    operator_dfs_filtered = {}

    for operator in sql_operators: 
        print(f"operator: {operator}")
        
        operator_df = operator_dfs[operator]
        print(f"operator count in squall dataset: {operator_df.shape[0]}")

        # Check matching ids with validation set
        id_matches_val = val_df[val_df['nt'].isin(operator_df['nt'])]
        operator_df_filtered = val_df[val_df['nt'].isin(operator_df['nt'])]
        operator_dfs_filtered[operator] = operator_df_filtered
            
        print(f"Number of matches in validation set: {len(id_matches_val)}")
        
    columns_to_drop = ['tbl', 'utterance_lower', 'nt']  # Columns to drop

    for operator in sql_operators: 
        filtered_df = operator_dfs_filtered[operator].drop(columns=columns_to_drop, errors='ignore')
        filtered_df.to_csv(f'raw_dataset/wtq/data/{operator}_filtered_split-1-dev.tsv', sep='\t', index=False)
        
    # Function to filter examples
    def filter_examples_by_ids(example_file, id_list, output_file):
        filtered_lines = []
        
        with open(example_file, 'r') as file:
            for line in file:
                if any(f"(id {id_})" in line for id_ in id_list):
                    filtered_lines.append(line)
        
        # Write the filtered lines to the new file
        with open(output_file, 'w') as file:
            file.writelines(filtered_lines)
        
        print(f"Filtered examples saved to {output_file}")

    original_examples_file = 'raw_dataset/wtq/data/random-split-1-dev.examples'

    for operator, operator_df in operator_dfs_filtered.items():
        id_list = operator_df['nt'].tolist()  
        #output_file = f"filtered_datasets/{operator}_filtered_split-1-dev.examples" 
        output_file = f'raw_dataset/wtq/data/{operator}_filtered_split-1-dev.examples'
        
        filter_examples_by_ids(original_examples_file, id_list, output_file)

if __name__ == "__main__":
    main()