import os
import requests

# List of dataset URLs and corresponding filenames
datasets = {
     "MMStar": "https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv",
     "MMU_TEST": "https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv",
     "MathVista_MINI": "https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv",
     "AI2D_TEST": "https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv",
     "ChartQA_TEST": "https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv",
     "OCRBench": "https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv",
     "TextVQA_VAL": "https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv",
     "RealWorldQA": "https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv",
     "MathVerse_MINI": "https://opencompass.openxlab.space/utils/VLMEval/MathVerse_MINI.tsv",
     "MathVision_MINI":  "https://opencompass.openxlab.space/utils/VLMEval/MathVision_MINI.tsv",
     "ScienceQA_TEST":  "https://opencompass.openxlab.space/utils/VLMEval/ScienceQA_TEST.tsv"
}

# Directory to save the datasets
save_dir = "LMUData"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Function to download datasets
def download_dataset(url, save_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Download each dataset
for dataset_name, url in datasets.items():
    file_path = os.path.join(save_dir, f"{dataset_name}.tsv")
    download_dataset(url, file_path)