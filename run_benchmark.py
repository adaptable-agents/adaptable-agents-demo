from datetime import datetime
import json
import os
import sys
import argparse
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the adaptable-agents-python-package to the path
package_path = Path(__file__).parent.parent / "adaptable-agents-python-package"
sys.path.insert(0, str(package_path))

from adaptable_agents import AdaptableOpenAIClient, CheatsheetConfig  # noqa: E402
from utils.evaluation import eval_for_GameOf24  # noqa: E402
from utils.extractor import extract_answer  # noqa: E402
from utils.execute_code import extract_and_run_python_code  # noqa: E402
from utils.logger import setup_logging, logger  # noqa: E402

PREDEFINED_PROMPTS = {
    "system": """# GENERATOR (PROBLEM SOLVER)

Instruction: You are an expert problem-solving assistant tasked with analyzing and solving various questions using your expertise and any provided context. Each task will include:
1. A specific question or problem to solve
2. Any relevant context or reference materials that may be provided

---

## 1. ANALYSIS & STRATEGY

- Carefully analyze the question and any provided context before starting
- Search for and identify any applicable patterns, strategies, or examples that may be relevant
- Create a structured approach to solving the problem at hand
- Review and document any limitations in the available information

## 2. SOLUTION DEVELOPMENT

- Present your solution using clear, logical steps that others can follow and review
- Explain your reasoning and methodology before presenting final conclusions
- Provide detailed explanations for each step of the process
- Check and verify all assumptions and intermediate calculations

## 3. PROGRAMMING TASKS

When coding is required:
- Write clean, efficient Python code
- Follow the strict code formatting and execution protocol (always use the Python code formatting block; furthermore, after the code block, always explicitly request execution by appending: "EXECUTE CODE!"):
  ```python
  # Your code here
  ```
  EXECUTE CODE!

- All required imports and dependencies should be clearly declared at the top of your code
- Include clear inline comments to explain any complex programming logic
- Perform result validation after executing your code
- Apply optimization techniques when applicable
- The code should be completely self-contained without external file dependencies--it should be ready to be executed right away
- Do not include any placeholders, system-specific paths, or hard-coded local paths
- Feel free to use standard and widely-used pip packages
- Opt for alternative methods if errors persist during execution
- Exclude local paths and engine-specific settings (e.g., avoid configurations like chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish"))

## 4. FINAL ANSWER FORMAT

ALWAYS present your final answer in the following format:

FINAL ANSWER:
<answer>
(final answer)
</answer>

N.B. Make sure that the final answer is properly wrapped inside the <answer> block.

* For multiple-choice questions: Only provide the letter choice (e.g., (A))
* For numerical answers: Only provide the final number (e.g., 42)
* For other types of answers, including free-response answers: Provide the complete final answer

Example:
Q: What is the meaning of life?
A: [...]
FINAL ANSWER:
<answer>
42
</answer>

""",
    "GameOf24": """
Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24.

Example:
Input: 4, 7, 8, 8
Output: (7 - (8 / 8)) * 4 = 24

""",
}


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run GameOf24 benchmark with Adaptable Agents"
    )

    # Task name
    parser.add_argument(
        "--task", type=str, default="GameOf24", help="Task name (default: GameOf24)"
    )

    # Model name
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )

    # Adaptable Agents API configuration
    parser.add_argument(
        "--adaptable_api_key",
        type=str,
        default=os.getenv("ADAPTABLE_API_KEY", "default-api-key"),
        help="Adaptable Agents API key",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Adaptable Agents API base URL",
    )
    parser.add_argument(
        "--memory_scope_path",
        type=str,
        default="gameof24/demo",
        help="Memory scope path (default: gameof24/demo)",
    )

    # Cheatsheet configuration
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for cheatsheet retrieval (default: 0.8)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=5,
        help="Maximum number of items in cheatsheet (default: 5)",
    )

    # Additional model-related arguments
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )

    # Save path arguments
    parser.add_argument(
        "--save_directory",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--max_n_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--no_shuffle", action="store_true", help="Disable dataset shuffling"
    )

    # Summarization configuration
    parser.add_argument(
        "--summarize_input",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Whether to summarize inputs before storage (true/false). "
        "If not provided, uses default configuration.",
    )

    # Code execution configuration
    parser.add_argument(
        "--allow_code_execution",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to allow code execution locally after receiving response (true/false, default: true)",
    )

    return parser.parse_args()


def read_file(file_path: str) -> str:
    """
    Read the file and return the content.
    """
    with open(file_path, "r") as file:
        return file.read()


def write_jsonl(file_path, data):
    """
    Save the outputs to a file.
    """
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")


def main(args):
    """
    Main function to run the benchmark using Adaptable Agents package.

    Args:
        args: Parsed command line arguments (argparse.Namespace)
    """
    # Set up logging
    logs_dir = setup_logging()
    logger.info(f"Benchmark logging initialized. Logs directory: {logs_dir}")
    logger.info("=" * 80)
    logger.info("Starting benchmark run")
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Memory scope path: {args.memory_scope_path}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"Max items: {args.max_items}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(
        f"Max samples: {args.max_n_samples if args.max_n_samples > 0 else 'all'}"
    )
    logger.info(f"Code execution: {args.allow_code_execution}")
    logger.info("=" * 80)

    # Load the dataset
    if args.task not in PREDEFINED_PROMPTS:
        raise ValueError(
            f"Task {args.task} is not recognized. Only GameOf24 is supported."
        )

    logger.info(f"Loading dataset: turingmachine/meta-prompting, task: {args.task}")
    dataset = load_dataset("turingmachine/meta-prompting")
    dataset = dataset[args.task]
    logger.info(f"Dataset loaded. Total examples: {len(dataset)}")

    # Initialize the Adaptable OpenAI client
    logger.info("Initializing Adaptable OpenAI client...")
    cheatsheet_config = CheatsheetConfig(
        similarity_threshold=args.similarity_threshold,
        max_items=args.max_items,
    )
    logger.debug(
        f"Cheatsheet config: similarity_threshold={args.similarity_threshold}, max_items={args.max_items}"
    )

    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable.\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )

    # Convert summarize_input string to bool if provided
    summarize_input = None
    if args.summarize_input is not None:
        summarize_input = args.summarize_input.lower() == "true"
        logger.info(f"Input summarization: {summarize_input}")

    # Convert allow_code_execution string to bool
    allow_code_execution = args.allow_code_execution.lower() == "true"

    client = AdaptableOpenAIClient(
        adaptable_api_key=args.adaptable_api_key,
        openai_api_key=openai_api_key,
        api_base_url=args.api_base_url,
        memory_scope_path=args.memory_scope_path,
        cheatsheet_config=cheatsheet_config,
        auto_store_memories=True,  # Automatically store memories after each generation
        summarize_input=summarize_input,
    )
    logger.info("Adaptable OpenAI client initialized successfully")

    # Create save path
    time_stamp = datetime.today().strftime("%Y-%m-%d-%H-%M")
    save_path_name = (
        f"{args.save_directory}/{args.task}/{args.model_name}_{time_stamp}.jsonl"
    )
    dir_path = os.path.dirname(save_path_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save the arguments
    save_param_path = save_path_name.replace(".jsonl", "_params.json")
    with open(save_param_path, "w") as file:
        # Convert args to dict
        args_dict = vars(args)
        json.dump(args_dict, file, indent=4)

    # Shuffle the dataset if needed
    if not args.no_shuffle:
        dataset = dataset.shuffle(seed=10)

    # Initialize outputs
    outputs = []
    correct_so_far = 0
    total_so_far = 0

    # Iterate over the dataset
    logger.info("Starting to process examples...")
    for idx, example in enumerate(dataset):
        original_input = example["input"]
        original_target = example.get("target", "")

        # Format the input with the task prompt
        task_prompt = PREDEFINED_PROMPTS[args.task]
        input_text = f"{task_prompt}\n\nQuestion #{idx+1}:\n{original_input}"

        # Log example start
        logger.info("=" * 80)
        logger.info(
            f"Example {idx+1}/{len(dataset) if args.max_n_samples <= 0 else args.max_n_samples}"
        )
        logger.info(f"Raw input: {original_input}")
        logger.info(f"Target: {original_target}")
        logger.debug(f"Full input text length: {len(input_text)} characters")

        # Print progress
        print(f"### Example {idx+1} ###")
        print(f"Input: {original_input}")

        # Fetch cheatsheet manually to log it (the client will fetch it again, but this allows us to log it)
        try:
            logger.info("Fetching cheatsheet...")
            cheatsheet = client.adaptable_agent.get_cheatsheet(input_text)
            if cheatsheet:
                logger.info(
                    f"Cheatsheet retrieved successfully (length: {len(cheatsheet)} characters)"
                )
                logger.debug(
                    f"Cheatsheet content (first 500 chars): {cheatsheet[:500]}..."
                )
            else:
                logger.warning(
                    "No cheatsheet retrieved (may be first run or no similar memories found)"
                )
        except Exception as e:
            logger.warning(f"Error fetching cheatsheet for logging: {str(e)}")
            cheatsheet = None

        # Use AdaptableOpenAIClient - it automatically:
        # 1. Fetches cheatsheet based on input_text
        # 2. Enhances the prompt with cheatsheet
        # 3. Calls OpenAI API
        # 4. Stores the memory automatically
        try:
            logger.info("Making API call...")
            logger.debug(
                f"API call parameters: model={args.model_name}, temperature={args.temperature}, max_tokens={args.max_tokens}"
            )
            logger.info(f"Input text sent to the adaptable agent client: {input_text}")
            system_prompt = PREDEFINED_PROMPTS["system"]
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            # Extract the initial output from the response
            output_text = response.choices[0].message.content or ""
            logger.info(f"Output text: {output_text}")

            # Log API response details
            logger.info("API call completed successfully")
            logger.debug(f"Response length: {len(output_text)} characters")
            if hasattr(response, "usage"):
                usage = response.usage
                logger.info(
                    f"Token usage: prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}, total_tokens={usage.total_tokens}"
                )
            logger.debug(f"Raw output (first 500 chars): {output_text[:500]}...")

            # Handle code execution locally if enabled
            code_execution_flag = "EXECUTE CODE!"
            if allow_code_execution and code_execution_flag in output_text:
                logger.info(
                    "Code execution flag detected, extracting and executing code..."
                )
                pre_code_execution_flag = output_text.split(code_execution_flag)[
                    0
                ].strip()
                # Check if output ends with code block marker
                if pre_code_execution_flag.endswith("```"):
                    output_prefix = pre_code_execution_flag
                    logger.debug(
                        f"Extracting code from output prefix (length: {len(output_prefix)})"
                    )
                    executed_code = extract_and_run_python_code(output_prefix)
                    if executed_code:
                        executed_code = executed_code.strip()
                        # Append executed code to output for evaluation
                        output_text = f"{output_text}\n\n{executed_code}".strip()
                        logger.info(
                            f"Code executed successfully. Output length: {len(executed_code)} characters"
                        )
                        logger.debug(f"Code execution output: {executed_code[:200]}...")
                    else:
                        logger.warning("Code execution returned no output")
                else:
                    logger.warning(
                        "Code execution flag found but output doesn't end with code block marker"
                    )
            else:
                logger.debug(
                    "No code execution flag detected or code execution disabled"
                )

            # Extract the final answer from the response
            logger.info("Extracting final answer from response...")
            final_answer = extract_answer(output_text)
            logger.info(f"Extracted answer: {final_answer}")
            if final_answer == "No final answer found":
                logger.warning("Failed to extract final answer from response")
                logger.debug(f"Full output for debugging: {output_text}")

            # Evaluate the result
            logger.info("Evaluating result...")
            result = eval_for_GameOf24(original_input, final_answer)

            if result:
                correct_so_far += 1
                logger.info("✓ CORRECT - Answer evaluation passed")
            else:
                logger.warning("✗ INCORRECT - Answer evaluation failed")
                logger.debug(
                    f"Evaluation details: input={original_input}, extracted_answer={final_answer}"
                )
            total_so_far += 1

            accuracy = (correct_so_far / total_so_far) * 100
            logger.info(
                f"Current accuracy: {correct_so_far}/{total_so_far} = {accuracy:.2f}%"
            )

            # Store output
            outputs.append(
                {
                    "input": input_text,
                    "raw_input": original_input,
                    "target": original_target,
                    "output": output_text,
                    "final_answer": final_answer,
                    "correct": result,
                    "example_number": idx + 1,
                }
            )

            print(f"Final answer: {final_answer}")
            print(f"Target: {original_target}")
            print(f"Correct: {result}")
            print(f"---- Correct so far: {correct_so_far}/{total_so_far}")
            print("###" * 50)

            # Log memory storage status (memory is stored automatically by the client)
            logger.debug("Memory storage: automatic (handled by AdaptableOpenAIClient)")

            # Save after each example
            logger.debug(f"Saving results to: {save_path_name}")
            write_jsonl(save_path_name, outputs)

        except Exception as e:
            logger.error(f"Error processing example {idx+1}: {str(e)}", exc_info=True)
            print(f"Error processing example {idx+1}: {str(e)}")
            outputs.append(
                {
                    "input": input_text,
                    "raw_input": original_input,
                    "target": original_target,
                    "error": str(e),
                    "example_number": idx + 1,
                }
            )
            total_so_far += 1
            write_jsonl(save_path_name, outputs)
            continue

        # Break if max_n_samples is reached
        if args.max_n_samples > 0 and idx == args.max_n_samples - 1:
            break

    # Final save
    logger.info("Saving final results...")
    write_jsonl(save_path_name, outputs)

    # Print summary
    final_accuracy = (correct_so_far / total_so_far * 100) if total_so_far > 0 else 0.0
    logger.info("=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"Total examples processed: {total_so_far}")
    logger.info(f"Correct answers: {correct_so_far}")
    logger.info(f"Final accuracy: {final_accuracy:.2f}%")
    logger.info(f"Results saved to: {save_path_name}")
    logger.info(f"Logs saved to: {logs_dir}")
    logger.info("=" * 80)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total examples: {total_so_far}")
    print(f"Correct: {correct_so_far}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    print(f"Results saved to: {save_path_name}")
    print(f"Logs saved to: {logs_dir}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
