cd /workspace
git clone --recursive https://github.com/ROCm/TransformerEngine.git
export NVTE_USE_HIPBLASLT=1
export NVTE_FRAMEWORK=pytorch
export PYTORCH_ROCM_ARCH=gfx942
cd TransformerEngine && pip install .
cd /workspace/llm-train-bench