# Adaptable Agents Demo

This demo demonstrates how to use the [Adaptable Agents Python package](../adaptable-agents-python-package/) with the [Adaptable Agents API server](../adaptable-agents/) to run benchmarks on GameOf24 and SWEBench tasks.

## Performance Comparison

The following table compares agent performance with and without Adaptable Agents on two benchmark tasks:

| Task | Without Adaptable Agents | With Adaptable Agents |
|------|-------------------------|----------------------|
| GameOf24 | 12.0% | 99.0% |
| SWEBench | 39.2% | 52.2% |
| BFCL | 76.58% | 86.57% |


## Quick Start Example

Using Adaptable Agents is incredibly simple. Just wrap your existing LLM client and it automatically learns from past interactions. **Both OpenAI and Anthropic are supported!**

### OpenAI Example

```python
from adaptable_agents import AdaptableOpenAIClient
from openai import OpenAI

# Simply wrap your existing OpenAI client
openai_client = OpenAI()  # Uses OPENAI_API_KEY from environment or your existing client
client = AdaptableOpenAIClient(
    adaptable_api_key="your-adaptable-api-key",
    api_base_url="https://api.adaptable-agents.com",
    openai_client=openai_client,
    memory_scope_path="my-project/task-name",
)

# Enable adaptable agents (True by default, but explicit for clarity)
client.enable_adaptable_agents = True

# Use it exactly like the OpenAI client - no code changes needed!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "apply arithmetic operators to convert these numbers to 24: 4, 7, 8, 8"}]
)

print(response.choices[0].message.content)
# The client automatically learns from each interaction to improve future responses
```

### Anthropic Example

```python
from adaptable_agents import AdaptableAnthropicClient
from anthropic import Anthropic

# Simply wrap your existing Anthropic client
anthropic_client = Anthropic()  # Uses ANTHROPIC_API_KEY from environment or your existing client
client = AdaptableAnthropicClient(
    adaptable_api_key="your-adaptable-api-key",
    api_base_url="https://api.adaptable-agents.com",
    anthropic_client=anthropic_client,
    memory_scope_path="my-project/task-name",
)

# Enable adaptable agents (True by default, but explicit for clarity)  
client.enable_adaptable_agents = True

# Use it exactly like the Anthropic client!
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "apply arithmetic operators to convert these numbers to 24: 4, 7, 8, 8"}]
)

print(response.content[0].text)
```

### Direct Pass-Through Mode

**Want to use it as a regular LLM client?** Just set the property to `False`:

```python
# Direct pass-through mode - no interception, no learning
client = AdaptableOpenAIClient(
    adaptable_api_key="your-adaptable-api-key",
    openai_api_key="your-openai-api-key",
)

# Disable adaptable agents for direct LLM calls, no modification
client.enable_adaptable_agents = False

# Calls go directly to OpenAI without any interception
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it! The same client works for both modes - you control when to learn and when to just use the LLM directly.

## Overview

Adaptable Agents automatically improves your LLM's performance by learning from past interactions. Simply wrap your existing OpenAI or Anthropic client and it will continuously get better over time.

## Prerequisites

1. **Adaptable Agents API Key and Base URL**: You will be provided with your API key and base URL for the cloud-hosted Adaptable Agents API server.

2. **LLM API Key**: Set your OpenAI or Anthropic API key as an environment variable:
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

3. **Python 3.8+**: Make sure you have Python 3.8 or higher

## Quick Start Guide

This guide will help you get the demo running quickly.

### Step 1: Start the Adaptable Agents API Server

In a separate terminal, start the server:

```bash
cd ../adaptable-agents
python -m src.api.main
```

The server should start on `http://localhost:8000`. Keep this terminal running.

### Step 2: Install Dependencies

```bash
cd adaptable-agents-demo
pip install -r requirements.txt
```

The adaptable-agents package is automatically loaded from the sibling folder - no separate installation needed!

### Step 3: Configure Your API Keys

Create a `.env` file in the project root and add your API keys:

```bash
# Create .env file
touch .env
```

Edit the `.env` file and add your API keys:

```bash
OPENAI_API_KEY=your-openai-api-key-here
ADAPTABLE_API_KEY=your-adaptable-api-key
API_BASE_URL=http://localhost:8000
```

You can get your API keys from:
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

### Step 4: Run the Demo

You can run evaluations using either the Makefile (recommended) or directly with Python scripts.

**Using Makefile (Recommended):**

```bash
# Run Game of 24 evaluation
make gameof24-eval

# Run SWE-bench evaluation
make swebench-eval

# Submit SWE-bench results
make swebench-submit

# Run browser-based task evaluation
make browser-eval

# See all available commands
make help
```

**Using Python Scripts Directly:**

```bash
# Run with default settings (processes all examples)
python run_benchmark.py

# Or run with a limited number of examples for testing
python run_benchmark.py --max_n_samples 5
```

### What Happens

1. The script loads the dataset (GameOf24, SWEBench, or browser tasks)
2. For each example:
   - Fetches a cheatsheet from the API (based on similar past experiences)
   - Enhances the prompt with the cheatsheet
   - Calls the LLM to generate a solution
   - Stores the input/output as a memory in the API
   - Evaluates the result
3. Results are saved to `results/` directory

### Expected Output

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

## Installation

Simply install the demo dependencies. The adaptable-agents package is automatically loaded from the sibling folder:

```bash
pip install -r requirements.txt
```

**Note**: The script automatically adds the `../adaptable-agents-python-package` directory to the Python path, so no separate installation is needed.

## Configuration

1. **Create a `.env` file** in the project root and add your API keys:
   ```bash
   # Create .env file
   touch .env
   
   # Add your API keys (you will be provided with Adaptable Agents credentials):
   # Use either OpenAI or Anthropic (or both)
   OPENAI_API_KEY=your-openai-api-key-here
   # OR
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   
   ADAPTABLE_API_KEY=your-adaptable-api-key
   API_BASE_URL=https://api.adaptable-agents.com
   ```


   You can get your API keys from:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/

The script will automatically load these values from the `.env` file using `python-dotenv`.

## Using Makefile

The Makefile provides convenient commands to run evaluations. This is the recommended way to run benchmarks.

### Available Commands

```bash
# Run Game of 24 evaluation benchmark
make gameof24-eval

# Run SWE-bench evaluation
make swebench-eval

# Submit SWE-bench results to sb-cli
make swebench-submit

# Run browser-based task evaluation (Playwright)
make browser-eval

# Show help and all available commands
make help
```

### Makefile Configuration

The Makefile targets use the following default configurations:

- **gameof24-eval**: Uses `gpt-4o` model, processes 20 samples, with similarity threshold 0.8
- **swebench-eval**: Uses `gpt-5` model, processes 10 samples from SWE-bench lite dev split
- **browser-eval**: Runs 15 multistep GitHub tasks with increasing complexity

All Makefile commands assume the API server is running at `http://localhost:8000`.

## Usage

This demo includes four benchmarks: **GameOf24**, **SWEBench**, **Browser-Use**, and **Browser-Playwright**.

You can run them using the Makefile (see above) or directly with Python scripts as shown below.

### GameOf24 Benchmark

Run the GameOf24 benchmark with default settings:

```bash
python run_benchmark.py
```

Or with custom configuration:

```bash
python run_benchmark.py \
  --adaptable_api_key "your-api-key" \
  --api_base_url "https://api.adaptable-agents.com" \
  --memory_scope_path "gameof24/demo" \
  --model_name "gpt-4o-mini" \
  --max_n_samples 10 \
  --similarity_threshold 0.8 \
  --max_items 5 \
  --summarize_input true \
  --allow_code_execution true \
  --max_depth_num_rounds 3
```

### SWEBench Benchmark

Run the SWEBench benchmark:

```bash
python run_swebench.py \
  --subset lite \
  --split dev \
  --model-name gpt-4o-mini \
  --memory-scope-path swebench-lite/demo \
  --similarity-threshold 0.8 \
  --max-items 10 \
  --max-n-samples 10 \
  --output results/swebench-adaptable
```

### Browser-Use Demo

This demo compares browser automation performance with and without Adaptable Agents using the [browser-use](https://github.com/browser-use/browser-use) library. It runs the same task (finding GitHub repo stars) using both approaches and shows performance improvements.

**Prerequisites:**
- Install Chromium browser: `uvx browser-use install` (or follow [browser-use installation guide](https://docs.browser-use.com/))

**Run the demo:**

```bash
python demo_browser_use.py
```

The demo will:
1. Run the task with a standard OpenAI client (without adaptable agents)
2. Run the same task with AdaptableOpenAIClient (with adaptable agents)
3. Display a comparison showing success rate, duration, and token usage

**Note:** The adaptable agent learns from each run, so performance improves over time as it builds up relevant context from past experiences. Run the demo multiple times to see the improvement!

**Environment Variables Required:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ADAPTABLE_API_KEY`: Your Adaptable Agents API key (required for adaptable agent comparison)
- `API_BASE_URL`: Adaptable Agents API base URL (default: `http://localhost:8000`)

### Browser-Playwright Demo

This demo compares browser automation performance with and without Adaptable Agents using [Playwright](https://playwright.dev/python/) directly. Unlike the browser-use demo, this version uses Playwright for browser automation and `AdaptableOpenAIClient` directly (similar to `run_benchmark.py` and `run_swebench.py`). It runs the same task (finding GitHub repo stars) using both approaches and shows performance improvements.

**Prerequisites:**
- Install Playwright and Chromium browser:
  ```bash
  pip install playwright
  playwright install chromium
  ```

**Run the demo:**
```bash
python demo_browser_playwright.py
```

The demo will:
1. Run the task with a standard OpenAI client (without adaptable agents)
2. Run the same task with AdaptableOpenAIClient (with adaptable agents)
3. Display a comparison showing success rate, duration, steps taken, and token usage

**Note:** The adaptable agent learns from each run, so performance improves over time as it builds up relevant context from past experiences. Run the demo multiple times to see the improvement!

**Environment Variables Required:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ADAPTABLE_API_KEY`: Your Adaptable Agents API key (required for adaptable agent comparison)
- `API_BASE_URL`: Adaptable Agents API base URL (default: `http://localhost:8000`)

### Arguments

**Task Configuration:**
- `--task`: Task name (default: "GameOf24", only GameOf24 is supported)
- `--model_name`: Model to use - supports OpenAI (e.g., "gpt-4o-mini", "gpt-4") or Anthropic (e.g., "claude-3-opus-20240229") models (default: "gpt-4o-mini")
- `--max_n_samples`: Maximum number of samples to process (-1 for all, default: -1)
- `--no_shuffle`: Disable dataset shuffling (default: False)

**Adaptable Agents API Configuration:**
- `--adaptable_api_key`: API key for Adaptable Agents API (provided to you, or from `.env`)
- `--api_base_url`: Base URL of Adaptable Agents API (provided to you, or from `.env`)
- `--memory_scope_path`: Memory scope path for organizing memories (default: "gameof24/demo")

**Learning Configuration:**
- `--similarity_threshold`: Similarity threshold for retrieving relevant past experiences (default: 0.8)
- `--max_items`: Maximum number of past experiences to use (default: 5)
- `--summarize_input`: Whether to summarize inputs before storage (true/false, optional)

**Model Generation Configuration:**
- `--max_tokens`: Maximum tokens for generation (default: 2048)
- `--temperature`: Temperature for generation (default: 0.0)
- `--max_depth_num_rounds`: Maximum number of refinement rounds to get final answer in correct format (default: 3)

**Code Execution Configuration:**
- `--allow_code_execution`: Whether to allow code execution locally after receiving response (true/false, default: true)

**Adaptable Agents Control:**
- `--enable_adaptable_agents`: Whether to enable automatic learning (true/false, default: true). If false, the client works as a standard LLM client without learning capabilities.

**Output Configuration:**
- `--save_directory`: Directory to save results (default: "results")

## Troubleshooting

### Server Connection Error
Make sure the API server is running on port 8000. You can start it with:
```bash
cd ../adaptable-agents
python -m src.api.main
```

### OpenAI API Key Error
Make sure you've set the `OPENAI_API_KEY` environment variable in your `.env` file.

### Module Not Found
Make sure you've installed the demo requirements:
```bash
pip install -r requirements.txt
```

The adaptable-agents package is automatically loaded from the sibling folder - no separate installation needed!
