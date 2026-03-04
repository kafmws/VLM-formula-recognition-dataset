CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model /root/dev/ckpts \
    --tensor-parallel-size 1 \
    --max-model-len 3072 \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-num-seqs 32 \
    --enable-chunked-prefill


python -m vllm.entrypoints.openai.api_server \
    --model /root/dev/ckpts \
    --tensor-parallel-size 1 \
    --max-model-len 3072 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --trust-remote-code \
    --quantization fp8 \
    --dtype bfloat16 \
    --max-num-seqs 32


INTERNVL_IMAGE_SIZE=448 swift sample \
    --model_type internvl3 \
    --model /root/dev/ckpts/latex_v0_5_norm_aug_ft4 \
    --local_repo_path /root/dev/ckpts/latex_v0_5_norm_aug_ft4 \
    --dataset /root/dev/data/datav0_5/train_norm.jsonl \
    --output_dir /root/dev/data/grpo/v0 \
    --sampler_engine vllm \
    --num_return_sequences 8 \
    --temperature 1.0 \
    --max_new_tokens 2048