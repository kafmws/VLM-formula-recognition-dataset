  python /root/dev/baseline/eval.py \
      --model-path /root/dev/baseline/swift_output/SFT-InternVL3-1B-lora/v8-20260213-060909/checkpoint-3000-merged \
      --input-dir /root/dev/baseline/data/samples_test \
      --output-dir ./results \
      --report-path ./evaluation_report.txt \
      --model-type vl