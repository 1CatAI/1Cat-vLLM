# 四模型七卡部署命令（2026-10-25）

按服务器保存的最终部署参数展开：Qwen GGUF、Qwen NVFP4、GLM GGUF、GLM NVFP4。适用机器为 Ubuntu `192.168.1.17`，登录用户 `zhangwei`，1Cat-vLLM 七卡异构 PP7 × TP1。

命令中的模型路径、Python 路径、JSON 参数、GPU 顺序全部直接写全；没有 shell 变量引用、占位符、读取 `.env` 或读取启动 manifest 的步骤。`env` 后的 CUDA/NCCL/VLLM 设置是程序运行需要的环境配置，值也全部写死，无需自行定义变量。

| 版本 | 每路输入＋输出总上限 | 最大同时执行数 | 草稿 | 图片上限 | GPU 顺序（物理编号） | 主模型层数分配 |
|---|---:|---:|---|---:|---|---|
| Qwen GGUF | 215040（210K） | 8 | MTP4，独立转换草稿目录 | 4 | 2,0,1,4,5,6,3 | 8,5,5,8,8,2,12 |
| Qwen NVFP4 | 215040（210K） | 8 | 原生 MTP4 | 4 | 2,0,1,4,5,6,3 | 8,5,5,8,8,2,12 |
| GLM GGUF | 163840（160K） | 3 | GGUF 原生 MTP4 | 8 | 2,0,1,6,4,5,3 | 9,4,4,5,7,7,9 |
| GLM NVFP4 | 163840（160K） | 1 | 原生 MTP4 | 8 | 2,0,1,4,5,6,3 | 8,5,5,6,6,6,9 |

表中是配置上限，不等于所有请求都能同时占满窗口。Qwen 的历史八路长短混合测试与当前 GLM NVFP4 的 160000-token 实测记录分别保存在相应归档中。GLM NVFP4 的目录虽名为 DFlash2，本命令实际启用的是 MTP4；尚未上线的独立 DFlash2 不混入已部署命令。

仓库中的独立脚本位于 [examples/online_serving/heterogeneous_pp7](../../examples/online_serving/heterogeneous_pp7/)，GLM NVFP4 的修复与验证说明见 [部署报告](glm_nvfp4_pp7_20261025.md)。

## 使用方式

在 Ubuntu 的 Bash 终端执行，四个完整代码块任选一个。每段开头会停止已有的四种模型服务，避免争抢七张卡和同一端口，然后以前台方式启动所选模型。日志显示在当前终端，Ctrl+C 停止。GPU6 按最新要求统一为 100W、最高 1050 MHz；历史部分 GGUF 压测用的是 950 MHz，不能直接当作当前频率下的性能结果。

四套服务均监听 `127.0.0.1:8001`，已有 `qwen-lan-proxy.service` 提供局域网入口 `http://192.168.1.17:8100/v1`。这些手动命令不会改变开机默认模型，当前开机默认仍为 GLM NVFP4。

四份独立 `.sh` 脚本与本文代码块相同。本地位于 `E:\workspaceDev03\archives\four-model-deployment-commands-20261025`；Ubuntu 副本位于 `/home/zhangwei/1cat-deploy/commands-20261025`。例如运行当前 GLM NVFP4：

```bash
bash /home/zhangwei/1cat-deploy/commands-20261025/glm-nvfp4.sh
```

所有命令依赖本机已经适配的 1Cat 源码及原生扩展，不能直接换成官方 pip vLLM。源码目录是 `/media/zhangwei/Sata02/llm-deploy/1cat-multiarch-6f3557c/glm-qwen-publish-20261024`；基线提交 `2d9737c0f7ad331cff272ede78a75a0c1649a3b5` 加本地 NVFP4 适配。GGUF 的 `/home/zhangwei/models` 路径已核验仍有效，实际通过符号链接访问 Sata02 上的文件；NVFP4 主权重在 SSD。

## Qwen 3.8 Flash Next GGUF · UD-Q4_K_XL

```bash
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
```

## Qwen 3.8 Flash Next NVFP4

```bash
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
```

## GLM 5.3 Flash GGUF · UD-IQ4_XS

```bash
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
```

## GLM 5.3 Flash Uncensored NVFP4 · 原生 MTP4

```bash
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
```

## 启动后的检查

在另一个终端执行：

```bash
curl -fsS http://192.168.1.17:8100/health
curl -fsS http://192.168.1.17:8100/v1/models
nvidia-smi
```

对应 API 模型 ID：Qwen GGUF 为 `qwen3.8-flash-next`（另有 `qwen3.8-flash-next-pp7`、`qwen3.8-flash-next-mtp4`）；Qwen NVFP4 为 `qwen3.8-flash-next-nvfp4`；GLM GGUF 为 `glm-5.3-flash`（另有 `glm-5.3-flash-pp7`）；GLM NVFP4 为 `glm-5.3-flash-nvfp4`（兼容 `glm-5.3-flash`）。OMP 当前默认的 GLM 别名可用于两种 GLM；换到 Qwen 后需选择对应 Qwen 模型。

手动进程停止后，恢复当前 GLM NVFP4 后台服务：

```bash
sudo systemctl start glm-nvfp4-pp7.service
```

## 归档与核验

本次只生成文档和脚本，未执行切换命令或重启模型。四份命令的 CLI 参数和环境设置与服务器保存的 manifest 逐项一致；独立脚本已做 Bash 语法检查。模型、草稿、分词器、生成配置、视觉文件路径及 GGUF 分片数量已检查。GLM NVFP4 是当前运行且已完成 160000-token、双图、工具调用验证的版本；其余三份是保留的部署配置，本次没有重新进行 GPU 回归。

原始 manifest 快照、路径核对与命令一致性记录同存于本地脚本目录。若以后修改这些配置，本文保存的字面命令不会自动随之变化。
