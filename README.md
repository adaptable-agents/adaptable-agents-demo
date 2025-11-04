# Adaptable Agents Demo - GameOf24 Benchmark

This demo demonstrates how to use the [Adaptable Agents Python package](../adaptable-agents-python-package/) with the [Adaptable Agents API server](../adaptable-agents/) to run a benchmark on the GameOf24 task.

## Overview

This demo shows how the Adaptable Agents system works end-to-end:
1. **Cheatsheet Retrieval**: Automatically fetches relevant past experiences from the API
2. **Prompt Enhancement**: Enhances prompts with retrieved cheatsheets
3. **LLM Generation**: Uses OpenAI to generate responses
4. **Memory Storage**: Automatically stores successful interactions for future learning

## Prerequisites

1. **Adaptable Agents API Server**: The server must be running at `http://localhost:8000` (or configure `api_base_url`)
   - See [adaptable-agents](../adaptable-agents/) for setup instructions
   - The server should be started and accessible

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

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** and add your OpenAI API key:
   ```bash
   # Open .env in your editor
   # Set your OpenAI API key:
   OPENAI_API_KEY=your-openai-api-key-here
   ```

   You can get your API key from: https://platform.openai.com/api-keys

3. **Optional**: Configure Adaptable Agents API settings in `.env`:
   ```bash
   ADAPTABLE_API_KEY=default-api-key
   API_BASE_URL=http://localhost:8000
   ```

The script will automatically load these values from the `.env` file.

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
  --api_base_url "http://localhost:8000" \
  --memory_scope_path "gameof24/demo" \
  --model_name "gpt-4o-mini" \
  --max_n_samples 10 \
  --similarity_threshold 0.8 \
  --max_items 5
```

### Arguments

- `--task`: Task name (default: "GameOf24", only GameOf24 is supported)
- `--model_name`: OpenAI model to use (default: "gpt-4o-mini")
- `--adaptable_api_key`: API key for Adaptable Agents API (default: "default-api-key")
- `--api_base_url`: Base URL of Adaptable Agents API (default: "http://localhost:8000")
- `--memory_scope_path`: Memory scope path for organizing memories (default: "gameof24/demo")
- `--similarity_threshold`: Similarity threshold for cheatsheet retrieval (default: 0.8)
- `--max_items`: Maximum number of items in cheatsheet (default: 5)
- `--max_tokens`: Maximum tokens for generation (default: 2048)
- `--temperature`: Temperature for generation (default: 0.0)
- `--save_directory`: Directory to save results (default: "results")
- `--max_n_samples`: Maximum number of samples to process (-1 for all, default: -1)
- `--no_shuffle`: Disable dataset shuffling (default: False)

## How It Works

1. **Initialization**: The script creates an `AdaptableOpenAIClient` instance that:
   - Connects to the Adaptable Agents API server
   - Configures cheatsheet retrieval settings
   - Sets up automatic memory storage

2. **For Each Example**:
   - **Cheatsheet Retrieval**: The client automatically fetches a cheatsheet based on the input text
   - **Prompt Enhancement**: The input prompt is enhanced with the retrieved cheatsheet
   - **LLM Call**: The enhanced prompt is sent to OpenAI
   - **Memory Storage**: The input/output pair is automatically stored in the API
   - **Evaluation**: The response is evaluated against the ground truth

3. **Results**: Results are saved to `results/GameOf24/` with timestamps

## Output Format

Results are saved as JSONL files with the following structure:

```json
{
  "input": "Full input text with task prompt",
  "raw_input": "Original input numbers",
  "target": "Target answer",
  "output": "Full model output",
  "final_answer": "Extracted final answer",
  "correct": true,
  "example_number": 1
}
```

## Example Output

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
Total examples: 10
Correct: 8
Accuracy: 80.00%
Results saved to: results/GameOf24/gpt-4o-mini_2024-01-15-14-30.jsonl
==================================================
```

## Integration Points

This demo showcases the integration between:

1. **Adaptable Agents Python Package** (`../adaptable-agents-python-package/`):
   - `AdaptableOpenAIClient`: Wraps OpenAI client with cheatsheet integration
   - `CheatsheetConfig`: Configuration for cheatsheet generation
   - Automatic memory storage after each generation

2. **Adaptable Agents API Server** (`../adaptable-agents/`):
   - `/api/v1/dynamic-cheatsheet/generate`: Generates cheatsheets from stored memories
   - `/api/v1/memories`: Stores and retrieves memories
   - Memory scope system for organizing memories

## Troubleshooting

### Server Not Running
```
Error: Connection refused
```
**Solution**: Make sure the Adaptable Agents API server is running:
```bash
cd ../adaptable-agents
python -m src.api.main
```

### OpenAI API Key Missing
```
Error: OpenAI API key not found
```
**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key"
```

### Package Not Found
```
ModuleNotFoundError: No module named 'adaptable_agents'
```
**Solution**: Install the package:
```bash
cd ../adaptable-agents-python-package
pip install -e .
```

## Next Steps

- Experiment with different `memory_scope_path` values to organize memories
- Adjust `similarity_threshold` and `max_items` to see how cheatsheet quality changes
- Try different models (e.g., `gpt-4`, `gpt-4o`)
- Monitor the API server logs to see memory storage and cheatsheet generation

## License

MIT License