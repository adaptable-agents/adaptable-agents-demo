"""
Run SWE-bench_Lite evaluation with AdaptableAgents integration.

This script evaluates SWE-bench_Lite instances using mini-swe-agent infrastructure
but with AdaptableAgents wrapper for context-aware LLM calls.
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the adaptable-agents-python-package to the path
package_path = Path(__file__).parent.parent / "adaptable-agents-python-package"
sys.path.insert(0, str(package_path))

# Add mini-swe-agent to the path (adjust if needed)
miniswe_path = Path(__file__).parent.parent / "mini-swe-agent"
if miniswe_path.exists():
    sys.path.insert(0, str(miniswe_path / "src"))

# Try to import mini-swe-agent components first
try:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.run.extra.swebench import (
        get_sb_environment,
        get_swebench_docker_image_name,
    )
    from minisweagent.run.utils.save import save_traj
    from minisweagent.utils.log import add_file_handler as add_miniswe_file_handler
    from minisweagent.config import builtin_config_dir
    import yaml
except ImportError as e:
    print(
        f"Failed to import mini-swe-agent components: {e}. "
        "Please ensure mini-swe-agent is installed or in the Python path."
    )
    raise

from utils.adaptable_model import AdaptableModel, AdaptableModelConfig  # noqa: E402
from utils.logger import setup_logging, logger  # noqa: E402

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench_Lite evaluation with AdaptableAgents"
    )

    # Dataset arguments
    parser.add_argument(
        "--subset",
        type=str,
        default="lite",
        help="SWEBench subset to use (default: lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="",
        help="Slice specification (e.g., '0:5' for first 5 instances)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="Filter instance IDs by regex",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle instances",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Redo existing instances",
    )
    parser.add_argument(
        "--max-n-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all, default: -1)",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (default: from OPENAI_API_KEY env var)",
    )

    # Adaptable Agents API configuration
    parser.add_argument(
        "--adaptable-api-key",
        type=str,
        default=os.getenv("ADAPTABLE_API_KEY", "default-api-key"),
        help="Adaptable Agents API key",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Adaptable Agents API base URL",
    )
    parser.add_argument(
        "--memory-scope-path",
        type=str,
        default="swebench-lite/demo",
        help="Memory scope path (default: swebench-lite/demo)",
    )

    # Cheatsheet configuration
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for cheatsheet retrieval (default: 0.8)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=5,
        help="Maximum number of items in cheatsheet (default: 5)",
    )

    # Adaptable agents enable/disable
    parser.add_argument(
        "--enable-adaptable-agents",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to enable adaptable agents functionality (true/false, default: true)",
    )

    # Summarization configuration
    parser.add_argument(
        "--summarize-input",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Whether to summarize inputs before storage (true/false). "
        "If not provided, uses default configuration.",
    )

    # Environment configuration
    parser.add_argument(
        "--environment-class",
        type=str,
        default="docker",
        help="Environment type to use (default: docker)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to agent config file (default: uses builtin swebench.yaml)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )

    return parser.parse_args()


def filter_instances(instances, filter_spec="", slice_spec="", shuffle=False):
    """Filter and slice a list of SWEBench instances."""
    import random
    import re

    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)

    before_filter = len(instances)
    if filter_spec:
        instances = [
            instance
            for instance in instances
            if re.match(filter_spec, instance["instance_id"])
        ]
        if (after_filter := len(instances)) != before_filter:
            logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")

    return instances


def update_preds_file(
    output_path: Path, instance_id: str, model_name: str, result: str
):
    """Update the output JSON file with results from a single instance."""
    output_data = {}
    if output_path.exists():
        output_data = json.loads(output_path.read_text())
    output_data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }
    output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    output_data = json.loads((output_path.read_text()))
    if instance_id in output_data:
        del output_data[instance_id]
        output_path.write_text(json.dumps(output_data, indent=2))


def process_instance(
    instance: dict,
    output_dir: Path,
    args,
    agent_config: dict,
):
    """Process a single SWEBench instance."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Avoid inconsistent state if something here fails
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    traj_file = instance_dir / f"{instance_id}.traj.json"
    if traj_file.exists():
        traj_file.unlink()

    # Convert summarize_input string to bool if provided
    summarize_input = None
    if args.summarize_input is not None:
        summarize_input = args.summarize_input.lower() == "true"

    # Convert enable_adaptable_agents string to bool
    enable_adaptable_agents = args.enable_adaptable_agents.lower() == "true"

    # Initialize AdaptableModel
    model = AdaptableModel(
        config_class=AdaptableModelConfig,
        model_name=args.model_name,
        adaptable_api_key=args.adaptable_api_key,
        openai_api_key=args.openai_api_key,
        api_base_url=args.api_base_url,
        memory_scope_path=args.memory_scope_path,
        similarity_threshold=args.similarity_threshold,
        max_items=args.max_items,
        auto_store_memories=True,
        summarize_input=summarize_input,
        enable_adaptable_agents=enable_adaptable_agents,
    )

    task = instance["problem_statement"]

    agent = None
    extra_info = None

    try:
        # Get environment configuration from agent_config
        env_config = agent_config.setdefault("environment", {})
        env_config["environment_class"] = args.environment_class
        image_name = get_swebench_docker_image_name(instance)
        if env_config["environment_class"] == "docker":
            env_config["image"] = image_name
        elif env_config["environment_class"] == "singularity":
            env_config["image"] = "docker://" + image_name

        env = get_sb_environment(agent_config, instance)

        # Create agent with config
        agent = DefaultAgent(model, env, **agent_config.get("agent", {}))

        logger.info(f"Processing instance {instance_id}...")
        exit_status, result = agent.run(task)

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Save trajectory
        save_traj(
            agent,
            instance_dir / f"{instance_id}.traj.json",
            exit_status=exit_status,
            result=result,
            extra_info=extra_info,
            instance_id=instance_id,
            print_fct=logger.info,
        )
        # Update predictions file
        update_preds_file(
            output_dir / "preds.json", instance_id, args.model_name, result
        )
        logger.info(f"Completed instance {instance_id}: {exit_status}")


def main(args):
    """Main function to run SWE-bench_Lite evaluation with AdaptableAgents."""
    # Set up logging
    logs_dir = setup_logging()
    logger.info(f"SWE-bench evaluation logging initialized. Logs directory: {logs_dir}")
    logger.info("=" * 80)
    logger.info("Starting SWE-bench_Lite evaluation with AdaptableAgents")
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Memory scope path: {args.memory_scope_path}")
    logger.info(f"Similarity threshold: {args.similarity_threshold}")
    logger.info(f"Max items: {args.max_items}")
    logger.info(f"Enable adaptable agents: {args.enable_adaptable_agents}")
    logger.info(f"Environment class: {args.environment_class}")
    logger.info("=" * 80)

    # Load agent config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = builtin_config_dir / "extra" / "swebench.yaml"
    logger.info(f"Loading agent config from '{config_path}'")
    agent_config = yaml.safe_load(config_path.read_text())
    if args.environment_class:
        agent_config.setdefault("environment", {})["environment_class"] = (
            args.environment_class
        )

    # Load dataset
    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    logger.info(f"Loading dataset {dataset_path}, split {args.split}...")
    instances = list(load_dataset(dataset_path, split=args.split))
    logger.info(f"Loaded {len(instances)} instances")

    # Filter instances
    instances = filter_instances(
        instances,
        filter_spec=args.filter,
        slice_spec=args.slice,
        shuffle=args.shuffle,
    )

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")

    # Add file handler for mini-swe-agent logging
    add_miniswe_file_handler(output_path / "minisweagent.log")

    # Skip existing instances if not redoing
    if not args.redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(
            json.loads((output_path / "preds.json").read_text()).keys()
        )
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [
            instance
            for instance in instances
            if instance["instance_id"] not in existing_instances
        ]

    # Limit number of samples if specified
    if args.max_n_samples > 0:
        instances = instances[: args.max_n_samples]

    logger.info(f"Running on {len(instances)} instances...")

    # Process instances sequentially (can be parallelized later)
    for idx, instance in enumerate(instances):
        logger.info(
            f"Processing instance {idx+1}/{len(instances)}: {instance['instance_id']}"
        )
        try:
            process_instance(instance, output_path, args, agent_config)
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing instance: {e}", exc_info=True)
            continue

    # Final summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"Total instances processed: {len(instances)}")
    logger.info(f"Results saved to: {output_path / 'preds.json'}")
    logger.info(f"Logs saved to: {logs_dir}")
    logger.info("=" * 80)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total instances: {len(instances)}")
    print(f"Results saved to: {output_path / 'preds.json'}")
    print(f"Logs saved to: {logs_dir}")
    print("=" * 50)
    print(
        "\nTo evaluate results, use SWE-bench CLI:\n"
        f"python -m swebench.harness.run_evaluation \\\n"
        f"    --dataset_name {dataset_path} \\\n"
        f"    --split {args.split} \\\n"
        f"    --predictions_path {output_path / 'preds.json'} \\\n"
        f"    --max_workers 4 \\\n"
        f"    --run_id adaptable-agents-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
