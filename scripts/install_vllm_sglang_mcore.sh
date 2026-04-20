#!/bin/bash

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}

export MAX_JOBS=32

echo "1. install inference frameworks and pytorch they need"
# Install vLLM first so it pins torch. SGLang's sgl_kernel (incl. SM90/SM100 common_ops) must match that
# same libtorch; installing SGLang before vLLM leaves stale CUDA extensions and undefined c10::SymInt symbols.
pip install --no-cache-dir "vllm==0.11.0"
if [ $USE_SGLANG -eq 1 ]; then
    pip install "sglang[all]==0.5.9" --no-cache-dir && pip install torch-memory-saver --no-cache-dir
fi
# If sgl_kernel fails with undefined c10::SymInt* symbols, torch changed after SGLang was installed; run:
#   pip install --force-reinstall --no-cache-dir "sglang[all]==0.5.9"

echo "2. install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard 

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
# Prebuilt wheels exist only for specific torch minor + C++ ABI. vLLM often ships torch 2.9+ with
# cxx11 ABI enabled; installing cxx11abiFALSE against that torch fails at import (undefined c10::SymInt*).
pip uninstall -y flash-attn 2>/dev/null || true
TORCH_MM=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
PYTAG=$(python -c "import sys; v=sys.version_info; print(f'cp{v.major}{v.minor}-cp{v.major}{v.minor}')")
FA_ABI=$(python -c "import torch; print('cxx11abiTRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'cxx11abiFALSE')")
FA_WHL="flash_attn-2.8.1+cu12torch${TORCH_MM}${FA_ABI}-${PYTAG}-linux_x86_64.whl"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FA_WHL}"
FA_CODE=$(curl -sSIL -o /dev/null -w "%{http_code}" "${FA_URL}")
if [ "${FA_CODE}" != "200" ]; then
    FA_ABI_ALT=$([ "${FA_ABI}" = "cxx11abiTRUE" ] && echo "cxx11abiFALSE" || echo "cxx11abiTRUE")
    FA_WHL_ALT="flash_attn-2.8.1+cu12torch${TORCH_MM}${FA_ABI_ALT}-${PYTAG}-linux_x86_64.whl"
    FA_URL_ALT="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/${FA_WHL_ALT}"
    FA_CODE_ALT=$(curl -sSIL -o /dev/null -w "%{http_code}" "${FA_URL_ALT}")
    if [ "${FA_CODE_ALT}" = "200" ]; then
        FA_WHL=${FA_WHL_ALT}
        FA_URL=${FA_URL_ALT}
        FA_ABI=${FA_ABI_ALT}
        FA_CODE=200
        echo "Using alternate flash-attn wheel (${FA_WHL})."
    fi
fi
if [ "${FA_CODE}" = "200" ]; then
    echo "Installing prebuilt flash-attn for torch ${TORCH_MM} ${FA_ABI} (${FA_WHL})"
    wget -nv "${FA_URL}" && pip install --no-cache-dir "${FA_WHL}" && rm -f "${FA_WHL}"
else
    echo "No prebuilt flash-attn 2.8.1 wheel for torch ${TORCH_MM} (${PYTAG}). Skipping flash-attn."
    echo "Options: (1) Use actor_rollout_ref.model.override_config.attn_implementation=sdpa (see examples/sglang_multiturn/config/search_multiturn_grpo.yaml)"
    echo "          (2) Build from source: MAX_JOBS=8 pip install flash-attn==2.8.1 --no-build-isolation"
fi

# SGLang 0.5.9 depends on flashinfer_python==0.6.3 / flashinfer_cubin==0.6.3. Do not install
# flashinfer-python==0.3.1 here: it was an older pin that runs after sglang and replaces the
# compatible wheels, then `import sglang...` fails because 0.3.1 has no
# `trtllm_mxint4_block_scale_moe` in `flashinfer.fused_moe` (ImportError from compressed_tensors).
# Use --no-deps so this line does not pull a newer torch than vLLM/SGLang (see pip metadata on flashinfer-python>=0.6).
pip install --no-cache-dir --no-deps "flashinfer-cubin==0.6.3" "flashinfer-python==0.6.3"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

# Reduce Py3.12 shutdown noise: older multiprocess resource_tracker vs threading.RLock
pip install --upgrade "multiprocess>=0.70.18"


if [ $USE_MEGATRON -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    pip install nvidia-cudnn-cu12==9.10.2.21
fi

echo "Successfully installed all packages"
