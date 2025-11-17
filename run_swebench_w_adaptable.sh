python run_swebench.py \
    --subset lite \
    --split dev \
    --model-name gpt-5 \
    --memory-scope-path swebench-lite/demo \
    --similarity-threshold 0.8 \
    --max-items 10 \
    --max-n-samples 10 \
    --output results/swebench-adaptable


sb-cli submit swe-bench_lite dev --predictions_path results/swebench-adaptable/preds.json --run_id mini-swe-agent-dev-$(date +%s)