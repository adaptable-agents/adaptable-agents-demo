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

    # Adaptable agents enable/disable
    parser.add_argument(
        "--enable_adaptable_agents",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to enable adaptable agents functionality (true/false, default: true). "
        "If false, calls are passed directly to the LLM without interception.",
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

    # Iterative refinement configuration
    parser.add_argument(
        "--max_depth_num_rounds",
        type=int,
        default=3,
        help="Maximum number of refinement rounds to get final answer in correct format (default: 3)",
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


def generate_with_refinement(
    client,
    system_prompt: str,
    initial_messages: list,
    extract_answer_func,
    allow_code_execution: bool,
    code_execution_flag: str,
    extract_and_run_python_code_func,
    max_depth_num_rounds: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    logger,
    current_depth: int = 1,
    final_output: str = "",
):
    """
    Generate response with iterative refinement to ensure final answer is in correct format.

    Args:
        client: AdaptableOpenAIClient instance
        system_prompt: System prompt for the model
        initial_messages: Initial conversation messages
        extract_answer_func: Function to extract final answer from response
        allow_code_execution: Whether to allow code execution
        code_execution_flag: Flag to trigger code execution
        extract_and_run_python_code_func: Function to extract and run Python code
        max_depth_num_rounds: Maximum number of refinement rounds
        model_name: Model name for API calls
        temperature: Temperature for generation
        max_tokens: Maximum tokens for generation
        logger: Logger instance
        current_depth: Current depth of recursion (default: 1)
        final_output: Accumulated output so far (default: "")

    Returns:
        tuple: (final_output_text, final_answer)
    """
    # Build messages for this round
    messages = [{"role": "system", "content": system_prompt}] + initial_messages

    # Make API call
    logger.debug(f"Making API call (round {current_depth}/{max_depth_num_rounds})...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Extract the output from the response
    output_text = response.choices[0].message.content or ""
    logger.debug(
        f"Round {current_depth} response received (length: {len(output_text)} chars)"
    )
    logger.info(f"Output text: {output_text}")

    # Handle code execution if needed
    pre_code_execution_flag = output_text.split(code_execution_flag)[0].strip()
    if (
        allow_code_execution
        and code_execution_flag in output_text
        and pre_code_execution_flag.endswith("```")
    ):
        logger.info(
            f"Code execution flag detected in round {current_depth}, executing code..."
        )
        output_prefix = pre_code_execution_flag
        executed_code = extract_and_run_python_code_func(output_prefix)
        if executed_code:
            executed_code = executed_code.strip()
            # Get any content after the code execution flag (e.g., FINAL ANSWER section)
            post_code_execution_flag = ""
            if code_execution_flag in output_text:
                parts = output_text.split(code_execution_flag, 1)
                if len(parts) > 1:
                    post_code_execution_flag = parts[1].strip()
            # Preserve the full output: prefix + flag + executed code + any content after flag
            if post_code_execution_flag:
                current_output = f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}\n\n{post_code_execution_flag}"
            else:
                current_output = (
                    f"{output_prefix}\n{code_execution_flag}\n\n{executed_code}"
                )
            logger.info(f"Code executed successfully in round {current_depth}")
            logger.debug(f"Code execution output: {executed_code}...")
        else:
            logger.warning("Code execution returned no output")
            current_output = output_text
    else:
        current_output = output_text

    # Accumulate output
    if final_output:
        final_output = f"{final_output}\n\n{current_output}".strip()
    else:
        final_output = current_output

    # Check if we have a valid final answer
    final_answer = extract_answer_func(final_output)
    has_valid_answer = final_answer != "No final answer found"
    logger.debug(
        f"Extracted answer: '{final_answer}' (has_valid_answer: {has_valid_answer})"
    )
    if not has_valid_answer:
        logger.debug(
            f"Full output for debugging (last 500 chars): {final_output[-500:]}"
        )

    # If we have a valid answer or exceeded max depth, return
    if has_valid_answer or current_depth > max_depth_num_rounds:
        if has_valid_answer:
            logger.info(f"Final answer found in round {current_depth}: {final_answer}")
        else:
            logger.warning(
                f"Max depth exceeded ({current_depth} > {max_depth_num_rounds}) without valid final answer"
            )
        return final_output, final_answer

    # Otherwise, make a follow-up call if we haven't exceeded max depth
    if current_depth <= max_depth_num_rounds:
        logger.info(
            f"No valid final answer found in round {current_depth}, making follow-up call..."
        )
        warning_txt = ""
        if current_depth == max_depth_num_rounds:
            warning_txt = " (This is the last round. No more code execution will be allowed. Please present your final solution now.)"

        new_messages = initial_messages + [
            {"role": "assistant", "content": current_output},
            {
                "role": "user",
                "content": f"Proceed with any additional steps required and provide the completed solution. If everything is already complete, type FINAL ANSWER and submit it in the expected format. If you are stuck, please try alternative methods to solve the problem and provide the final solution.{warning_txt}",
            },
        ]

        # Recursive call for next round
        return generate_with_refinement(
            client=client,
            system_prompt=system_prompt,
            initial_messages=new_messages,
            extract_answer_func=extract_answer_func,
            allow_code_execution=allow_code_execution,
            code_execution_flag=code_execution_flag,
            extract_and_run_python_code_func=extract_and_run_python_code_func,
            max_depth_num_rounds=max_depth_num_rounds,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            logger=logger,
            current_depth=current_depth + 1,
            final_output=final_output,
        )

    # Fallback: should not reach here, but return what we have
    return final_output, final_answer


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
    logger.info(f"Max depth rounds: {args.max_depth_num_rounds}")
    logger.info(f"Enable adaptable agents: {args.enable_adaptable_agents}")
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

    # Convert enable_adaptable_agents string to bool
    enable_adaptable_agents = args.enable_adaptable_agents.lower() == "true"
    logger.info(f"Adaptable agents enabled: {enable_adaptable_agents}")

    client = AdaptableOpenAIClient(
        adaptable_api_key=args.adaptable_api_key,
        openai_api_key=openai_api_key,
        api_base_url=args.api_base_url,
        memory_scope_path=args.memory_scope_path,
        cheatsheet_config=cheatsheet_config,
        auto_store_memories=True,  # Automatically store memories after each generation
        summarize_input=summarize_input,
        enable_adaptable_agents=enable_adaptable_agents,
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
        # Only if adaptable agents is enabled
        cheatsheet = None
        if enable_adaptable_agents:
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
        else:
            logger.info("Adaptable agents disabled - skipping cheatsheet fetch")

        # Use AdaptableOpenAIClient with iterative refinement - it automatically:
        # 1. Fetches cheatsheet based on input_text
        # 2. Enhances the prompt with cheatsheet
        # 3. Calls OpenAI API with iterative refinement to get final answer in correct format
        # 4. Stores the memory automatically
        try:
            logger.info("Starting iterative refinement process...")
            logger.debug(
                f"API call parameters: model={args.model_name}, temperature={args.temperature}, max_tokens={args.max_tokens}, max_depth={args.max_depth_num_rounds}"
            )
            logger.info(f"Input text sent to the adaptable agent client: {input_text}")
            system_prompt = PREDEFINED_PROMPTS["system"]
            code_execution_flag = "EXECUTE CODE!"

            # Use iterative refinement to get final answer in correct format
            initial_messages = [{"role": "user", "content": input_text}]
            output_text, final_answer = generate_with_refinement(
                client=client,
                system_prompt=system_prompt,
                initial_messages=initial_messages,
                extract_answer_func=extract_answer,
                allow_code_execution=allow_code_execution,
                code_execution_flag=code_execution_flag,
                extract_and_run_python_code_func=extract_and_run_python_code,
                max_depth_num_rounds=args.max_depth_num_rounds,
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                logger=logger,
            )

            logger.info(f"Final output text length: {len(output_text)} characters")
            logger.info(f"Extracted final answer: {final_answer}")
            if final_answer == "No final answer found":
                logger.warning(
                    "Failed to extract final answer after all refinement rounds"
                )
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
