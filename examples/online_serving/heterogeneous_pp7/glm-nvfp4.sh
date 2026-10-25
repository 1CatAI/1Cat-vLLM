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
  OMP_NUM_THREADS=8 \
  NCCL_P2P_DISABLE=1 \
  NCCL_CUMEM_ENABLE=0 \
  NCCL_CUMEM_HOST_ENABLE=0 \
  VLLM_SM70_FLASH_V100_DECODE_GRAPH_NO_COMPILE=1 \
  VLLM_PP_LAYER_PARTITION=8,5,5,6,6,6,9 \
  CUDA_HOME=/usr/local/cuda-12.8 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/glm-causal-fixes-20261023/moe-down-m40-production-source/.venv/bin:/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/source/.venv/bin:/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/source/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
  PYTHONPATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1 \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/zhangwei/models/GLM-5.3-Flash-UNCENSORED-NVFP4-DFlash2 \
  --served-model-name glm-5.3-flash-nvfp4 glm-5.3-flash \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 7 \
  --distributed-executor-backend mp \
  --dtype float16 \
  --attention-backend GLM5_SM70_SPARSE \
  --max-model-len 163840 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --pp-async-queue-depth 4 \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 5], "max_cudagraph_capture_size": 5}' \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4, "attention_backend": "GLM5_SM70_SPARSE", "draft_sample_method": "greedy", "use_local_argmax_reduction": true, "enforce_eager": false}' \
  --limit-mm-per-prompt '{"image": 8, "video": 1}' \
  --mm-processor-kwargs '{"max_image_tokens": 256}' \
  --media-io-kwargs '{"video": {"num_frames": 8}}' \
  --mm-encoder-attn-backend TORCH_SDPA \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --mixed-prefill-min-tokens 128
