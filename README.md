# Adaptable Agents Demo - GameOf24 Benchmark

This demo demonstrates how to use the [Adaptable Agents Python package](../adaptable-agents-python-package/) with the [Adaptable Agents API server](../adaptable-agents/) to run a benchmark on the GameOf24 task.

## Performance Comparison

The following table compares agent performance with and without Adaptable Agents on two benchmark tasks:

| Task | Without Adaptable Agents | With Adaptable Agents |
|------|-------------------------|----------------------|
| GameOf24 | 15.3% | 100.0% |
| SWEBench | 39.2% | 60.0% |


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
    enable_adaptable_agents=True
)

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
    enable_adaptable_agents=True
)

# Use it exactly like the Anthropic client!
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "apply arithmetic operators to convert these numbers to 24: 4, 7, 8, 8"}]
)

print(response.content[0].text)
```

### Direct Pass-Through Mode

**Want to use it as a regular LLM client?** Just set `enable_adaptable_agents=False`:

```python
# Direct pass-through mode - no interception, no learning
client = AdaptableOpenAIClient(
    adaptable_api_key="your-adaptable-api-key",
    openai_api_key="your-openai-api-key",
    enable_adaptable_agents=False  # Direct LLM calls, no modification
)

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

2. **OpenAI API Key**: Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Python 3.8+**: Make sure you have Python 3.8 or higher

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
   
   # Add your API keys (you will be provided with these):
   OPENAI_API_KEY=your-openai-api-key-here
   ADAPTABLE_API_KEY=your-adaptable-api-key
   API_BASE_URL=https://api.adaptable-agents.com
   ```

   You can get your OpenAI API key from: https://platform.openai.com/api-keys

The script will automatically load these values from the `.env` file using `python-dotenv`.

## Usage

### Basic Usage

Run the benchmark with default settings:

```bash
python run_benchmark.py
```

### Custom Configuration

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

### Arguments

**Task Configuration:**
- `--task`: Task name (default: "GameOf24", only GameOf24 is supported)
- `--model_name`: OpenAI model to use (default: "gpt-4o-mini")
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
