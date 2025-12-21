.PHONY: help gameof24-eval swebench-eval swebench-submit browser-eval

# Default target executed when no arguments are given to make.
all: help

######################
# EVALUATION SCRIPTS
######################

gameof24-eval:
	@echo "🎮 Running Game of 24 evaluation..."
	@python run_benchmark.py \
		--api_base_url "http://localhost:8000" \
		--memory_scope_path "gameof248745878783" \
		--model_name "gpt-4o" \
		--max_n_samples 20 \
		--similarity_threshold 0.8 \
		--max_items 5 \
		--summarize_input true \
		--allow_code_execution true

swebench-eval:
	@echo "🔧 Running SWE-bench evaluation..."
	@python run_swebench.py \
		--subset lite \
		--split dev \
		--model-name gpt-5 \
		--memory-scope-path swebench-lite/demo \
		--similarity-threshold 0.8 \
		--max-items 10 \
		--max-n-samples 10 \
		--output results/swebench-adaptable

swebench-submit:
	@echo "📤 Submitting SWE-bench results..."
	@sb-cli submit swe-bench_lite dev \
		--predictions_path results/swebench-adaptable/preds.json \
		--run_id mini-swe-agent-dev-$$(date +%s)

browser-eval:
	@echo "🌐 Running browser-based task evaluation..."
	@python demo_browser_playwright.py

######################
# HELP
######################

help:
	@echo '🚀 Adaptable Agents Demo - Commands'
	@echo '===================================='
	@echo ''
	@echo '📊 EVALUATION:'
	@echo '  make gameof24-eval                 - Run Game of 24 evaluation benchmark'
	@echo '  make swebench-eval                 - Run SWE-bench evaluation'
	@echo '  make swebench-submit               - Submit SWE-bench results to sb-cli'
	@echo '  make browser-eval                  - Run browser-based task evaluation (Playwright)'
	@echo ''
	@echo '💡 Usage:'
	@echo '  Ensure the API server is running at http://localhost:8000'
	@echo '  Then run: make gameof24-eval, make swebench-eval, or make browser-eval'

