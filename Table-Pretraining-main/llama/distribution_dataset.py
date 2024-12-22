import os
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Dict mapping dataset_name -> list of JSON paths.
# We merge train.json, valid.json, and test.json into a single set.
DATASET_PATHS = {
    "WikiSQL": [
        "dataset/wikisql/train.json",
        "dataset/wikisql/valid.json",
        "dataset/wikisql/test.json"
    ],
    "Spider": [
        "dataset/spider/train.json",
        "dataset/spider/valid.json",
        "dataset/spider/test.json"
    ],
    "WTQ": [
        "dataset/wtq/train.json",
        "dataset/wtq/valid.json",
        "dataset/wtq/test.json"
    ],
    # Add more datasets if needed
}

SAVE_DIR = "dataset/plots"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_CHARS = 5000  # Only used for filtering *when plotting*

def load_multiple_datasets(datasets_dict):
    """
    Given a dict of {dataset_name: [list_of_json_paths]},
    load each set of JSON files into a single Dataset (under the key 'train').
    Returns a dict of {dataset_name: loaded_dataset}
    """
    loaded_datasets = {}
    for dataset_name, json_paths in datasets_dict.items():
        print(f"Loading dataset '{dataset_name}' from:\n  {json_paths}")
        
        # Merging train, valid, test into a single "train" split
        ds = load_dataset(
            'json',
            data_files={'train': json_paths}
        )
        loaded_datasets[dataset_name] = ds['train']  # everything in ds['train']
    return loaded_datasets

def process_dataset(dataset, dataset_name):
    """
    Collect two distributions of character lengths:
    1) unfiltered_char_lengths: all examples (for stats).
    2) filtered_char_lengths: only examples <= MAX_CHARS (for plotting).

    Returns (unfiltered_char_lengths, filtered_char_lengths).
    """
    unfiltered_char_lengths = []
    filtered_char_lengths = []
    
    for example in tqdm(dataset, desc=f"Processing dataset ({dataset_name})", unit="example"):
        # If your dataset columns differ, adjust these keys
        input_text = example["input"]
        output_text = example["output"]
        
        total_chars = len(input_text) + len(output_text)
        
        # 1) Unfiltered distribution for statistics
        unfiltered_char_lengths.append(total_chars)

        # 2) Filtered distribution for plotting
        if total_chars <= MAX_CHARS:
            filtered_char_lengths.append(total_chars)
    
    return unfiltered_char_lengths, filtered_char_lengths

def print_unfiltered_statistics(dataset_name, unfiltered_lengths):
    """
    Prints various statistics (mean, median, etc.) for the unfiltered distribution.
    """
    if not unfiltered_lengths:
        print(f"{dataset_name}: No data available for statistics.")
        return
    
    arr = np.array(unfiltered_lengths)
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_val = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    pct90 = np.percentile(arr, 90)
    pct95 = np.percentile(arr, 95)

    print(f"----- Statistics for {dataset_name} (Unfiltered) -----")
    print(f"  Total Examples: {len(arr)}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val}")
    print(f"  Std Dev: {std_val:.2f}")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"  90th Percentile: {pct90}")
    print(f"  95th Percentile: {pct95}")
    print("-------------------------------------------------------\n")

def plot_line_distributions(distributions_dict):
    """
    Given a dictionary of {dataset_name: filtered_char_lengths},
    plot the line distribution (frequency vs. character length) for each dataset.
    """

    colors = ["#8338EC", "#4085F4", "#1CC549"]  # cycle if > 3 datasets
    line_styles = ["-", "-", "-", "-"]

    plt.figure(figsize=(10, 6))

    for i, (dataset_name, lengths) in enumerate(distributions_dict.items()):
        if not lengths:
            print(f"No valid examples (<= {MAX_CHARS} chars) in dataset '{dataset_name}'. Skipping plot.")
            continue

        # Bins of size 100 (adjust if needed)
        bins = np.arange(0, MAX_CHARS + 100, 100)
        counts, edges = np.histogram(lengths, bins=bins)
        
        frequencies = counts / np.sum(counts)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]

        plt.plot(
            bin_centers,
            frequencies,
            label=dataset_name,
            linestyle=linestyle,
            color=color
        )
        # Fill the area under the curve
        plt.fill_between(bin_centers, frequencies, alpha=0.2, color=color)

    plt.title(f"Character Length Distribution (Cutoff at {MAX_CHARS} chars)")
    plt.xlabel("Number of Characters (input + output)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(SAVE_DIR, "character_length_distributions.png")
    plt.savefig(output_path)
    plt.close()
    print(f"\nPlot saved to: {output_path}\n")

def main():
    # 1. Load multiple datasets (train, valid, test combined)
    datasets_dict = load_multiple_datasets(DATASET_PATHS)
    
    # 2. Process each dataset, get both unfiltered and filtered distributions
    distributions_for_plot = {}
    for dataset_name, dataset in datasets_dict.items():
        print(f"----- Processing dataset: {dataset_name} -----")
        unfiltered_lengths, filtered_lengths = process_dataset(dataset, dataset_name)

        # Print stats for unfiltered data
        print_unfiltered_statistics(dataset_name, unfiltered_lengths)

        # Store filtered distribution for plotting
        distributions_for_plot[dataset_name] = filtered_lengths
    
    # 3. Plot the distributions (filtered) for all datasets together
    plot_line_distributions(distributions_for_plot)

if __name__ == "__main__":
    main()
