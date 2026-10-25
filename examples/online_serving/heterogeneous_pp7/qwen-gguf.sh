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
  /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/zhangwei/models/Qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --served-model-name qwen3.8-flash-next qwen3.8-flash-next-pp7 qwen3.8-flash-next-mtp4 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --dtype float16 \
  --decode-context-parallel-size 1 \
  --attention-backend FLASH_ATTN_V100 \
  --quantization gguf \
  --load-format gguf \
  --max-model-len 215040 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4, "model": "/home/zhangwei/models/Qwen3.8-Flash-Next/MTP/hf-from-gguf-q8-pp-complete", "draft_load_config": {"load_format": "safetensors"}, "index_share_for_mtp_iteration": false}' \
  --kernel-config '{"sm70_mtp_q8_0_experts": true, "sm70_rmsnorm_gated_exact": false}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [5, 10, 15, 20, 25, 30, 35, 40]}' \
  --pipeline-parallel-size 7 \
  --distributed-executor-backend mp \
  --hf-config-path /home/zhangwei/models/Qwen3.8-Flash-Next/HF-multimodal-assets/gguf-compatible \
  --tokenizer /home/zhangwei/models/Qwen3.8-Flash-Next/HF-multimodal-assets/gguf-compatible \
  --limit-mm-per-prompt '{"image": 4, "video": 1}' \
  --mm-processor-kwargs '{"size": {"shortest_edge": 4096, "longest_edge": 262144}}' \
  --media-io-kwargs '{"video": {"num_frames": 8}}' \
  --mm-encoder-attn-backend TORCH_SDPA \
  --pp-async-queue-depth 4 \
  --additional-config '{"mtp_tail_warmup": true, "pp_decode_static_transfer": true, "pp_packed_feedback": true, "pp_transport_metrics_interval": 256}' \
  --mixed-prefill-step-latency-ms 0
