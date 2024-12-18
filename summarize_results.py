import os
import csv
import argparse

# Find result files ending with '_acc.csv' in the outputs directory for the given model
def find_result_files(model_name):
    result_files = []
    for file in os.listdir(f"outputs/{model_name}/"):
        if file.endswith("_acc.csv"):
            result_files.append(f"outputs/{model_name}/{file}")
    return result_files

# Create a file to store the evaluations on different datasets for this model
def create_summary_file(model_name):
    summary_file = open(f"outputs/{model_name}/{model_name}_summary.csv", "w")
    # Create columns for dataset name and overall accuracy
    summary_file.write("Dataset,Accuracy\n")
    return summary_file

# Write the results of the evaluation to the summary file
def write_summary(summary_file, dataset_name, accuracy):
    summary_file.write(f"{dataset_name},{accuracy}\n")

# Summarize the results of the evaluations
def summarize_results(model_name):
    result_files = find_result_files(model_name)

    # Check if the summary file exists, otherwise create it
    if not os.path.exists(f"{model_name}_summary.csv"):
        summary_file = create_summary_file(model_name)
    else:
        summary_file = open(f"{model_name}_summary.csv", "a")

    for result_file in result_files:
        # Get the dataset name from the file name, the part between the model name and '_acc.csv'
        dataset_name = result_file.split(model_name + "_")[1].split("_acc.csv")[0]

        # Read the accuracy from the result file (2 lines: header and data)
        with open(result_file, "r") as file:
            reader = csv.DictReader(file)
            row = next(reader)  # Read the first (and only) data row
            accuracy = row.get("Overall", None)

        # Write the accuracy to the summary file if found
        if accuracy is not None:
            write_summary(summary_file, dataset_name, accuracy)
        else:
            print(f"Warning: 'Overall' column not found in {result_file}")

    summary_file.close()
    print(f"Results summarized in {model_name}_summary.csv")

# Main function
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Summarize evaluation results for a given model.")
    parser.add_argument("model_name", type=str, help="Name of the model to process results for")
    args = parser.parse_args()

    # Call the summarize function with the provided model name
    summarize_results(args.model_name)

# Example usage: python summarize_results.py InternVL2-1B 