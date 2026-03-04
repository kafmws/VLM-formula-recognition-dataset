python /root/dev/data/newgenerator/data_generator.py \
    --input /root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl \
    --stats /root/dev/data/newgenerator/orignal.json \
    --output /root/dev/data/newgenerator/generated.jsonl \
    --num-samples 80000 \
    --workers 4 \
    --img-dir /root/dev/data/datav3/gimgs \
    --error-output error.jsonl \
    --top 10 \
    --strict-verify



python /root/dev/data/newgenerator/data_generator.py \
    --input /root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl \
    --stats /root/dev/data/newgenerator/orignal.json \
    --output /root/dev/data/newgenerator/generated.jsonl \
    --num-samples 300 \
    --top 10 \
    --strict-verify