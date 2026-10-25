#!/usr/bin/env bash
set -euo pipefail

# These four deployments share all seven GPUs and port 8001. Run one at a time.
sudo systemctl stop glm-nvfp4-pp7.service qwen-nvfp4-pp7.service glm-pp7.service qwen-pp7.service
sudo /usr/bin/nvidia-smi -i GPU-1cf29838-9742-0d6e-a5b6-92e6bbd0c4ce -pm 1
sudo /usr/bin/nvidia-smi -i GPU-1cf29838-9742-0d6e-a5b6-92e6bbd0c4ce -pl 100
sudo /usr/bin/nvidia-smi -i GPU-1cf29838-9742-0d6e-a5b6-92e6bbd0c4ce -lgc 135,1050
sudo systemctl start qwen-lan-proxy.service

cd /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024

/usr/bin/env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=2,0,1,4,5,6,3 \
  VLLM_PLE_CPU_OFFLOAD=1 \
  VLLM_PLE_DISK_OFFLOAD=1 \
  VLLM_SM70_QWEN38_HYBRID_PLE=0 \
  OMP_NUM_THREADS=8 \
  NCCL_P2P_DISABLE=1 \
  NCCL_CUMEM_ENABLE=0 \
  NCCL_CUMEM_HOST_ENABLE=0 \
  VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1 \
  VLLM_PP_LAYER_PARTITION=8,5,5,8,8,2,12 \
  CUDA_HOME=/usr/local/cuda-12.8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200 \
  PATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/glm-causal-fixes-20261023/moe-down-m40-production-source/.venv/bin:/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/source/.venv/bin:/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/source/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
  PYTHONPATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024 \
  /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/zhangwei/models/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen3.8-flash-next-nvfp4 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 7 \
  --distributed-executor-backend mp \
  --dtype float16 \
  --max-model-len 215040 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --pp-async-queue-depth 4 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --limit-mm-per-prompt '{"image": 4, "video": 1}' \
  --mm-encoder-attn-backend TORCH_SDPA \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --mixed-prefill-min-tokens 128
