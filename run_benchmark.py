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

from adaptable_agents import AdaptableOpenAIClient, CheatsheetConfig
from utils.evaluation import eval_for_GameOf24
from utils.extractor import extract_answer

PREDEFINED_PROMPTS = {
    "GameOf24": "Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24.",
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
    # Load the dataset
    if args.task not in PREDEFINED_PROMPTS:
        raise ValueError(
            f"Task {args.task} is not recognized. Only GameOf24 is supported."
        )

    dataset = load_dataset("turingmachine/meta-prompting")
    dataset = dataset[args.task]

    # Initialize the Adaptable OpenAI client
    cheatsheet_config = CheatsheetConfig(
        similarity_threshold=args.similarity_threshold,
        max_items=args.max_items,
    )

    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable.\n"
            "Get your API key from: https://platform.openai.com/api-keys"
        )

    client = AdaptableOpenAIClient(
        adaptable_api_key=args.adaptable_api_key,
        openai_api_key=openai_api_key,
        api_base_url=args.api_base_url,
        memory_scope_path=args.memory_scope_path,
        cheatsheet_config=cheatsheet_config,
        auto_store_memories=True,  # Automatically store memories after each generation
    )

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
    for idx, example in enumerate(dataset):
        original_input = example["input"]
        original_target = example.get("target", "")

        # Format the input with the task prompt
        task_prompt = PREDEFINED_PROMPTS[args.task]
        input_text = f"{task_prompt}\n\nQuestion #{idx+1}:\n{original_input}"

        # Print progress
        print(f"### Example {idx+1} ###")
        print(f"Input: {original_input}")

        # Use AdaptableOpenAIClient - it automatically:
        # 1. Fetches cheatsheet based on input_text
        # 2. Enhances the prompt with cheatsheet
        # 3. Calls OpenAI API
        # 4. Stores the memory automatically
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=[{"role": "user", "content": input_text}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            # Extract the final answer from the response
            output_text = response.choices[0].message.content or ""
            final_answer = extract_answer(output_text)

            # Evaluate the result
            result = eval_for_GameOf24(original_input, final_answer)

            if result:
                correct_so_far += 1
            total_so_far += 1

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

            # Save after each example
            write_jsonl(save_path_name, outputs)

        except Exception as e:
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
    write_jsonl(save_path_name, outputs)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total examples: {total_so_far}")
    print(f"Correct: {correct_so_far}")
    print(f"Accuracy: {correct_so_far / total_so_far * 100:.2f}%")
    print(f"Results saved to: {save_path_name}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
