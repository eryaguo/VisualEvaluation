#!/bin/bash
#SBATCH --job-name=vlm-evaluation
#SBATCH --container-writable
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/nanotron_pretrain_x86/nanotron_pretrain_x86.toml
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4


# Activate the virtual environment
cd $SCRATCH/vlm_eval_infer/
source vlmeval_env_extend/bin/activate
echo "Using Python: $(which python)"

# Navigate to the VLMEvalKit directory
cd VLMEvalKit
echo "Current directory: $(pwd)"

export LMUData="$SCRATCH/vlm_eval_infer/LMUData/"

export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

python -m pip install --no-index --find-links=$SCRATCH/vlm_eval_infer/package_caches -r requirements.txt

# Install VLMEvalKit in editable mode and update numpy
python -m pip install -e .


# Fix OpenCV ImportError by using headless version
python -m pip uninstall -y opencv-python
python -m pip install --no-index --find-links=$SCRATCH/vlm_eval_infer/package_caches opencv-python-headless

python -m pip uninstall -y transformers

python -m pip install --no-index --find-links=$SCRATCH/vlm_eval_infer/package_caches wheel
th

python -m pip install  --no-index --find-links=$SCRATCH/vlm_eval_infer/package_caches timm

cd $SCRATCH/vlm_eval_infer/package_caches/transformers
git checkout idefics3
python -m pip install --no-index --find-links=$SCRATCH/vlm_eval_infer/package_caches -q .
pip list | grep transformers

echo "Current directory: $(pwd)"
cd $SCRATCH/vlm_eval_infer/VLMEvalKit

# List files to confirm the installation
ls -l

Run the VLM evaluation
Python run.py \
  --data MMStar MMMU_TEST MathVista_MINI AI2D_TEST ChartQA_TEST OCRBench TextVQA_VAL RealWorldQA \
  --model Idefics3_8B_Llama3_from_cache \
  --verbose

# python run.py \
#   --data ScienceQA_TEST MMStar AI2D_TEST ChartQA_TEST OCRBench TextVQA_VAL RealWorldQA \
#   --model Idefics3_8B_Llama3_from_nanotron \
#   --verbose

# python run.py \
#   --data ScienceQA_TEST MMStar AI2D_TEST ChartQA_TEST OCRBench TextVQA_VAL RealWorldQA \
#   --model idefics2_8b_from_cache \
#   --verbose

python summarize_results.py idefics2_8b_from_cache

# Deactivate the virtual environment
deactivate