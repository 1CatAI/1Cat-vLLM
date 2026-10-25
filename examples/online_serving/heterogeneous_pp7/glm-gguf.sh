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
  CUDA_VISIBLE_DEVICES=GPU-174bb6e7-d49a-dcda-545f-04ec0bb96485,GPU-13fc1f56-25f5-e91d-7b85-ddefcab55668,GPU-c07e6021-3676-97d8-ca40-d3636b8cab90,GPU-1cf29838-9742-0d6e-a5b6-92e6bbd0c4ce,GPU-c4a75a41-3321-bfba-0473-02894d410410,GPU-e42052fe-6ef4-78c8-7259-37540ecdbc21,GPU-2ede7ca9-6965-fd0d-cecc-6bbb82183c57 \
  VLLM_PP_LAYER_PARTITION=9,4,4,5,7,7,9 \
  VLLM_USE_V2_MODEL_RUNNER=1 \
  OMP_NUM_THREADS=8 \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  NCCL_P2P_DISABLE=1 \
  NCCL_CUMEM_ENABLE=0 \
  NCCL_CUMEM_HOST_ENABLE=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  CUDA_HOME=/usr/local/cuda-12.8 \
  PYTHONPATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024 \
  PATH=/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/source/.venv/bin:/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-pp7/source/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin \
  VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1 \
  /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/zhangwei/models/GLM-5.3-Flash-UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  --served-model-name glm-5.3-flash glm-5.3-flash-pp7 \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 7 \
  --distributed-executor-backend mp \
  --dtype float16 \
  --attention-backend GLM5_SM70_SPARSE \
  --quantization gguf \
  --load-format gguf \
  --kernel-config '{"sm70_gguf":{"expert_storage":"original","expert_arena":true,"embedding_storage":"dense"},"canonical_prefill_topk_order":false,"gguf_moe_packed_down":true}' \
  --hf-config-path /home/zhangwei/models/GLM-5.3-Flash-UD-IQ4_XS/hf-gguf-assets \
  --tokenizer /home/zhangwei/models/GLM-5.3-Flash-UD-IQ4_XS/hf-gguf-assets \
  --max-model-len 163840 \
  --max-num-seqs 3 \
  --max-num-batched-tokens 768 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --reasoning-parser glm45 \
  --pp-async-queue-depth 2 \
  --additional-config '{"pp_decode_static_transfer": false, "pp_packed_feedback": false, "pp_transport_metrics_interval": 0, "verified_mtp_prefill_cache": true, "mtp_tail_warmup": true}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [5, 10, 15], "max_cudagraph_capture_size": 15}' \
  --num-gpu-blocks-override 228 \
  --generation-config /media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm53-gguf-mtp-pp7/generation-config-single-assistant \
  --limit-mm-per-prompt '{"image": 8, "video": 1}' \
  --mm-processor-kwargs '{"images_kwargs": {"max_image_tokens": 256}, "videos_kwargs": {"max_image_tokens": 1024}}' \
  --media-io-kwargs '{"video": {"num_frames": 8}}' \
  --mm-encoder-attn-backend TORCH_SDPA \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4, "model": "/home/zhangwei/models/GLM-5.3-Flash-UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf", "quantization": "gguf", "draft_load_config": {"load_format": "gguf"}, "attention_backend": "GLM5_SM70_SPARSE", "draft_sample_method": "greedy", "use_local_argmax_reduction": true, "enforce_eager": false}' \
  --long-prefill-token-threshold 128 \
  --long-prefill-token-threshold-adaptive \
  --pp-prefill-quota-compensation \
  --pp-decode-stagger
