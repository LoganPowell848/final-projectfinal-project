# NPS HPC & JUPYTER NOTEBOOK GROOT-DREAM

## Executive Summary
Aim - A runnable demonstration of GR00T-Dreams was
assembled. GR00T-Dreams is a research pipeline that converts
visual input and natural-language prompts into robot-action world
models and short videos (reasoning + vision-language planning). 
-Execution on Windows/Jupyter was prioritized to avoid
cluster/GPU requirements while still producing artifacts.
- The pipeline from NVIDIA Cosmos/GR00T was adapted so a
minimal example could be run on CPU. - Prompts were supplied
and a sample image was processed to produce demonstration
MP4.
- A full, CPU-only run path was produced end-to-end in Jupyter.
- Repository code (cosmos-predict2, GR00T-Dreams) was cloned
and organized. - Dependency gaps were resolved by installing
Python packages and by introducing small compatibility stubs.
- A playlist HTML viewer was created so outputs could be viewed
inside the notebook environment. - A compact report generator
(this cell) was added for documentation.
 - CUDA: NVIDIA GPU compute platform (not used here; CPU-only
was enforced). - Dot-Product Attention: similarity-based weighting
in Transformers (matrix product of queries and keys).
- FlashAttention: fast GPU attention kernel (not used in CPU mode).
   -Megatron-Core: distributed Transformer training/inference utilities
(imports satisfied; GPU features bypassed). - Transformer Engine
(TE): NVIDIA kernels/modules for high-performance Transformer
ops (CPU stubs used here). - Stubs: minimal Python stand-ins that
satisfy imports without GPU kernels
- A conda environment with PyTorch (CPU build) was used. - CUDA
usage was disabled to avoid GPU lookups and errors. - Heavy
GPU-only features (e.g., flash-attention) were not invoked.
-MP4 Latest observed:
gr1_14B_cpu_demo_20250926_114007.mp4
• - Diagnostic logs present: 6
• - Sample image present: Yes
- Inference was performed on CPU; speed and quality are reduced
compared to GPU.
• - Compatibility stubs were used; true GPU kernels were not
executed.

## Repository Layout
- `src/`  source code (entry points below)
- `configs/`  run-time configs (e.g., hyperparams, paths)
- `docs/final_presentation.pdf`  slides for final presentation
- `requirements.txt` or `environment.yml`  dependencies
- `logs/`  runtime logs (stdout/stderr)
- `README.md`  this quick guide + overview

## Quick Start
```bash
# LOGIN TO SSH WITH HAMMING USING MOBAXTERM THROUGH SSH
cd $WORK/$USER/datasets
# clone NVIDIA’s repo from your fork
git clone https://github.com/<your-username>/GR00T-Dreams.git
# CHANGE FROM HOME DIRECTORY TO /smallwork/YOUR.USER/mkdir
git clone https://github.com/NVIDIA/GR00T-Dreams.git
cd GR00T-Dreams
# MAKE SMALLWORK DIRECTORY
mkdir -p $SMALLWORK/$USER
# ACCESS SMALL WORK
cd /smallwork/your.username
# CREATE NEW ENVIROMENT
conda create -n projectenv
# INITIALIZE CONDA
conda init
# ACTIVATE YOUR ENVIROMENT
conda activate yourenviroment
# USE A NODE
srun --11 --pty bash
# SEE AVAILABLE SOFTWARE
module avail
# LOAD SOFTWARE SELECTED
module load lang/miniconda3/24.3.0


