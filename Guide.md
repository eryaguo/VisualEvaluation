Go through these steps to set up environment to implement VLMEvalKit on bristen.

# Setup for the bristen Cluster
Follow the instruction in the [bristen cluster setup](https://github.com/swiss-ai/documentation/blob/main/pages/setup_bristen.md) to have access to the bristen cluster.

# Set up Nvidia GPU Cloud (NGC) Access to use Nvidia Containers
Follow the instructions in the [NGC setup](https://github.com/swiss-ai/documentation/blob/main/pages/setup_ngc.md) to set up NGC access.

# Set up for the Nanotron Pretrain Container
We will use the existing nanotron container (nanotron_pretrain_x86) for visual evaluation. 
## Set up for Podman
To use Podman, we first need to configure some storage locations for it. Create the file `$HOME/.config/containers/storage.conf`:
```conf
[storage]
  driver = "overlay"
  runroot = "/dev/shm/$USER/runroot"
  graphroot = "/dev/shm/$USER/root"
 
[storage.options.overlay]
  mount_program = "/usr/bin/fuse-overlayfs-1.13"
```
## Setup an EDF 
We need to set up an EDF (Environment Definition File) which tells the Container Engine what container to load, where to mount it, and what plugins to load. Create a file ~/.edf/nanotron_pretrain.toml for the container engine. The EDF should look like this:
```toml
# image = "/store/swissai/a06/containers/nanotron_pretrain/latest/nanotron_pretrain.sqsh"
image = "/capstor/store/cscs/swissai/a06/containers/nanotron_pretrain_x86/nanotron_pretrain_x86.sqsh"
mounts = [
  "/capstor",
  "/iopsstor",
]


# mounts = [
#   "/capstor",
#   "/iopsstor",
#   "/store",
# ]

workdir = "/workspace/nanotron"

[env]
FI_CXI_DISABLE_HOST_REGISTER = "1"
FI_MR_CACHE_MONITOR = "userfaultfd"

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda12"
```
## Setup the Python Virtual Environment
We will need to install extra packages to make the visual evaluation work. Create a python virtual environment and download the required packages into our local cache folder since we don't have internet access on the compute node.
### Download the required packages for VLMEvalKit
```bash
[cluster][user@cluster-ln001 ~]$ cd $SCRATCH && mkdir -p vlm_eval && cd vlm_eval
[cluster][user@cluster-ln001 vlm_eval]$ python -m venv ./vlmeval_env
[cluster][user@cluster-ln001 vlm_eval]$ source ./vlmeval_env/bin/activate
```
Now we will install the required packages to package cache folder.
```bash
[cluster][user@cluster-ln001 vlm_eval]$ git clone https://github.com/open-compass/VLMEvalKit.git
[cluster][user@cluster-ln001 vlm_eval]$ cd VLMEvalKit
[cluster][user@cluster-ln001 VLMEvalKit]$ mkdir -p package_caches 
[cluster][user@cluster-ln001 VLMEvalKit]$ pip download -d $SCRATCH/vlm_eval/package_caches -r requirements.txt
```
### Download the transformers 4.45.0-dev and other required packages for idefics3
```bash
[cluster][user@cluster-ln001 vlm_eval]$ pip download opencv-python-headless wheel -d $SCRATCH/vlm_eval/package_caches
[cluster][user@cluster-ln001 vlm_eval]$ git clone https://github.com/huggingface/transformers.git
[cluster][user@cluster-ln001 vlm_eval]$ cd transformers
[cluster][user@cluster-ln001 transformers]$ git checkout idefics3
[cluster][user@cluster-ln001 transformers]$ pip download -d $SCRATCH/vlm_eval/package_caches .
```
### Add Custom Model Checkpoint into model config file: `config.py`
```bash
[cluster][user@cluster-ln001 vlm_eval]$ cd $SCRATCH/vlm_eval/VLMEvalKit/
[cluster][user@cluster-ln001 VLMEvalKit]$ cd vlmeval
[cluster][user@cluster-ln001 vlmeval]$ vi config.py
```
Then add the following content:
```python
idefics_series = {
    # add here the path to the model checkpoint
    'idefics2_8b_from_cache': partial(IDEFICS2, model_path='/path/to/your/model/checkpoint'),
    'Idefics3-8B-Llama3_from_cache': partial(IDEFICS2, model_path='/path/to/your/model/checkpoint'),
    'Idefics3_8B_Llama3_from_nanotron': partial(IDEFICS2, model_path='/path/to/your/model/checkpoint'),
}
```
# Prepare the script for results summarization
Create a file `summarize_results.py` to efficiently summarize the evaluation results for a given model.
# Prepare the SLURM script
We will use the following SLURM script to run the container on the bristen cluster, including setting up the virtual environment and running the visual evaluation tasks. Create a file `vlmeval.slurm` with the command:
```bash
[cluster][user@cluster-ln001 ~]$ cd $SCRATCH/vlm_eval && vi vlmeval.slurm
```
Create the following content in the file `vlmeval.slurm`:
```slurm
#!/bin/bash
#SBATCH --job-name=vlm-evaluation
#SBATCH --container-writable
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/nanotron_pretrain_x86/nanotron_pretrain_x86.toml
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4

# Activate the virtual environment
cd $SCRATCH/vlm_eval/
source vlmeval_env/bin/activate
echo "Using Python: $(which python)"

# Navigate to the VLMEvalKit directory
cd VLMEvalKit
echo "Current directory: $(pwd)"

export LMUData="$SCRATCH/vlm_eval/LMUData/"

export GIT_DISCOVERY_ACROSS_FILESYSTEM=1

python -m pip install --no-index --find-links=$SCRATCH/vlm_eval/package_caches -r requirements.txt

# Install VLMEvalKit in editable mode
python -m pip install -e .


# Fix OpenCV ImportError by using headless version
python -m pip uninstall -y opencv-python
python -m pip install --no-index --find-links=$SCRATCH/vlm_eval/package_caches opencv-python-headless

# Update transformers to the idefics3 branch
python -m pip uninstall -y transformers
python -m pip install --no-index --find-links=$SCRATCH/vlm_eval/package_caches wheel

cd $SCRATCH/vlm_eval/package_caches/transformers

git checkout idefics3
python -m pip install --no-index --find-links=$SCRATCH/vlm_eval/package_caches -q .
pip list | grep transformers

echo "Current directory: $(pwd)"
cd $SCRATCH/vlm_eval/VLMEvalKit

# List files to confirm the installation
ls -l

python run.py \
  --data ScienceQA_TEST MMStar AI2D_TEST ChartQA_TEST TextVQA_VAL RealWorldQA \
  --model Idefics3_8B_Llama3_from_nanotron \
  --verbose

python summarize_results.py Idefics3_8B_Llama3_from_nanotron

# Deactivate the virtual environment
deactivate
```
# Run the SLURM script for the Visual Evaluation
Run the SLURM script with the following command:
```bash
[cluster][user@cluster-ln001 ~]$ cd $SCRATCH/vlm_eval
[cluster][user@cluster-ln001 vlm_eval]$ sbatch vlmeval.slurm
```
# Internet Issues on the Compute Node
If you encounter internet issues on the compute node when running the VLM Evaluation, you can download the required packages on the login node with the following script: 
```bash
[cluster][user@cluster-ln001 ~]$ cd $SCRATCH/vlm_eval
[cluster][user@cluster-ln001 vlm_eval]$ vi download_testset.py 
[cluster][user@cluster-ln001 vlm_eval]$ python download_testset.py
```

