# Adaptable Agents Demo - GameOf24 Benchmark

This demo demonstrates how to use the [Adaptable Agents Python package](../adaptable-agents-python-package/) with the [Adaptable Agents API server](../adaptable-agents/) to run a benchmark on the GameOf24 task.

## Quick Start Example

Using Adaptable Agents is incredibly simple. Just wrap your existing LLM client and it automatically learns from past interactions:

```python
from adaptable_agents import AdaptableOpenAIClient

# Initialize the client - it works just like the regular OpenAI client
client = AdaptableOpenAIClient(
    adaptable_api_key="your-adaptable-api-key",
    openai_api_key="your-openai-api-key",
    memory_scope_path="my-project/task-name",
    enable_adaptable_agents=True  # Enable learning and cheatsheet retrieval
)

# Use it exactly like the OpenAI client - no code changes needed!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Solve: 4, 7, 8, 8 to get 24"}]
)

print(response.choices[0].message.content)
# The client automatically:
# 1. Fetches relevant past experiences as a cheatsheet
# 2. Enhances your prompt with the cheatsheet
# 3. Stores the interaction for future learning
```

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

1. **Create a `.env` file** in the project root and add your OpenAI API key:
   ```bash
   # Create .env file
   touch .env
   
   # Add your OpenAI API key:
   OPENAI_API_KEY=your-openai-api-key-here
   ```

   You can get your API key from: https://platform.openai.com/api-keys

2. **Optional**: Configure Adaptable Agents API settings in `.env`:
   ```bash
   ADAPTABLE_API_KEY=default-api-key
   API_BASE_URL=http://localhost:8000
   ```

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
  --api_base_url "http://localhost:8000" \
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
- `--adaptable_api_key`: API key for Adaptable Agents API (default: "default-api-key" or from `.env`)
- `--api_base_url`: Base URL of Adaptable Agents API (default: "http://localhost:8000" or from `.env`)
- `--memory_scope_path`: Memory scope path for organizing memories (default: "gameof24/demo")

**Cheatsheet Configuration:**
- `--similarity_threshold`: Similarity threshold for cheatsheet retrieval (default: 0.8)
- `--max_items`: Maximum number of items in cheatsheet (default: 5)
- `--summarize_input`: Whether to summarize inputs before storage (true/false, optional)

**Model Generation Configuration:**
- `--max_tokens`: Maximum tokens for generation (default: 2048)
- `--temperature`: Temperature for generation (default: 0.0)
- `--max_depth_num_rounds`: Maximum number of refinement rounds to get final answer in correct format (default: 3)

**Code Execution Configuration:**
- `--allow_code_execution`: Whether to allow code execution locally after receiving response (true/false, default: true)

**Adaptable Agents Control:**
- `--enable_adaptable_agents`: Whether to enable adaptable agents functionality (true/false, default: true). If false, calls are passed directly to the LLM without interception, cheatsheet retrieval, or memory storage.

**Output Configuration:**
- `--save_directory`: Directory to save results (default: "results")

## How It Works

1. **Initialization**: The script creates an `AdaptableOpenAIClient` instance that:
   - Connects to the Adaptable Agents API server
   - Configures cheatsheet retrieval settings
   - Sets up automatic memory storage
   - Initializes logging to a timestamped directory

2. **For Each Example**:
   - **Cheatsheet Retrieval**: The client automatically fetches a cheatsheet based on the input text from similar past experiences
   - **Prompt Enhancement**: The input prompt is enhanced with the retrieved cheatsheet
   - **Iterative Refinement**: The script uses an iterative refinement process:
     - Makes an initial LLM call with the enhanced prompt
     - Extracts and executes Python code if present (when `--allow_code_execution true`)
     - Checks if the final answer is in the correct format
     - Makes follow-up calls if needed (up to `--max_depth_num_rounds` rounds)
     - Ensures the final answer is properly formatted in `<answer>...</answer>` tags
   - **Memory Storage**: The input/output pair is automatically stored in the API after successful generation
   - **Evaluation**: The response is evaluated against the ground truth

3. **Results**: 
   - Results are saved to `results/GameOf24/` with timestamps as JSONL files
   - Parameters are saved alongside results as `*_params.json` files
   - Detailed logs are saved to `logs/{timestamp}/benchmark.log`

## Output Format

### Results Files

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

### Parameter Files

A `*_params.json` file is saved alongside each results file containing all command-line arguments used for that run:

```json
{
  "task": "GameOf24",
  "model_name": "gpt-4o-mini",
  "adaptable_api_key": "default-api-key",
  "api_base_url": "http://localhost:8000",
  "memory_scope_path": "gameof24/demo",
  "similarity_threshold": 0.8,
  "max_items": 5,
  "max_tokens": 2048,
  "temperature": 0.0,
  "save_directory": "results",
  "max_n_samples": -1,
  "no_shuffle": false,
  "summarize_input": null,
  "allow_code_execution": "true",
  "max_depth_num_rounds": 3
}
```

### Logging

Detailed logs are automatically saved to timestamped directories:
- **Location**: `logs/{YYYY-MM-DD_HH-MM-SS}/benchmark.log`
- **Content**: Includes debug information, API calls, cheatsheet retrieval, code execution, and evaluation results
- **Format**: Timestamped entries with log levels (DEBUG, INFO, WARNING, ERROR)

## Example Output

### Console Output

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
Logs saved to: logs/2024-01-15_14-30-45
==================================================
```

### Log File Output

The log file contains detailed information about each step:
- Cheatsheet retrieval status and content
- API call parameters and responses
- Code execution results (if applicable)
- Iterative refinement rounds
- Evaluation details

## Integration Points

This demo showcases the integration between:

1. **Adaptable Agents Python Package** (`../adaptable-agents-python-package/`):
   - `AdaptableOpenAIClient`: Wraps OpenAI client with cheatsheet integration
   - `CheatsheetConfig`: Configuration for cheatsheet generation
   - Automatic memory storage after each generation
   - Input summarization support (optional)

2. **Adaptable Agents API Server** (`../adaptable-agents/`):
   - `/api/v1/dynamic-cheatsheet/generate`: Generates cheatsheets from stored memories
   - `/api/v1/memories`: Stores and retrieves memories
   - Memory scope system for organizing memories

3. **Local Utilities** (`utils/`):
   - `evaluation.py`: GameOf24-specific evaluation logic
   - `extractor.py`: Extracts final answers from model responses
   - `execute_code.py`: Executes Python code extracted from responses
   - `logger.py`: Sets up timestamped logging system

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
**Solution**: The script automatically adds the package path, but if you encounter this error:
```bash
cd ../adaptable-agents-python-package
pip install -e .
```

### Code Execution Errors
If code execution fails, you can disable it:
```bash
python run_benchmark.py --allow_code_execution false
```

### Final Answer Format Issues
If the model isn't providing answers in the correct format, increase refinement rounds:
```bash
python run_benchmark.py --max_depth_num_rounds 5
```

## Next Steps

- **Experiment with memory organization**: Try different `memory_scope_path` values to organize memories by experiment or configuration
- **Tune cheatsheet retrieval**: Adjust `similarity_threshold` and `max_items` to see how cheatsheet quality affects performance
- **Try different models**: Test with `gpt-4`, `gpt-4o`, or other OpenAI models
- **Enable input summarization**: Use `--summarize_input true` to store more concise memories
- **Monitor logs**: Check the timestamped log files in `logs/` for detailed debugging information
- **Review parameters**: Check the `*_params.json` files to track which configurations work best
- **Monitor API server**: Watch the Adaptable Agents API server logs to see memory storage and cheatsheet generation in real-time

## License

MIT License