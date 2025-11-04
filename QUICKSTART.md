# Quick Start Guide

This guide will help you get the demo running quickly.

## Step 1: Start the Adaptable Agents API Server

In a separate terminal, start the server:

```bash
cd ../adaptable-agents
python -m src.api.main
```

The server should start on `http://localhost:8000`. Keep this terminal running.

## Step 2: Set Your OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Step 3: Install Dependencies

```bash
cd adaptable-agents-demo
pip install -r requirements.txt
```

The adaptable-agents package is automatically loaded from the sibling folder - no separate installation needed!

## Step 4: Run the Demo

Run with default settings (processes all examples):

```bash
python run_benchmark.py
```

Or run with a limited number of examples for testing:

```bash
python run_benchmark.py --max_n_samples 5
```

## What Happens

1. The script loads the GameOf24 dataset
2. For each example:
   - Fetches a cheatsheet from the API (based on similar past experiences)
   - Enhances the prompt with the cheatsheet
   - Calls OpenAI to generate a solution
   - Stores the input/output as a memory in the API
   - Evaluates the result
3. Results are saved to `results/GameOf24/`

## Expected Output

```
### Example 1 ###
Input: 4 7 8 8
Final answer: (7 - (8 / 8)) * 4
Target: 
Correct: True
---- Correct so far: 1/1
##################################################

==================================================
SUMMARY
Total examples: 5
Correct: 4
Accuracy: 80.00%
Results saved to: results/GameOf24/gpt-4o-mini_2024-01-15-14-30.jsonl
==================================================
```

## Troubleshooting

### Server Connection Error
Make sure the API server is running on port 8000.

### OpenAI API Key Error
Make sure you've set the `OPENAI_API_KEY` environment variable.

### Module Not Found
Make sure you've installed both:
- The adaptable-agents package (`pip install -e .` in adaptable-agents-python-package)
- The demo requirements (`pip install -r requirements.txt`)
