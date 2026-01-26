"""
Run LoCoMo dataset evaluation with Adaptable Agents using KG (AMEM) strategy.

This script:
1. Loads the LoCoMo dataset from data/locomo10.json
2. Stores conversation turns as memories in Adaptable Agents
3. Answers questions using KG strategy (AMEM) for context retrieval
4. Evaluates answers using comprehensive metrics
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the adaptable-agents-python-package to the path
package_path = Path(__file__).parent.parent / "adaptable-agents-python-package"
sys.path.insert(0, str(package_path))

# Add data directory to path for dataset loading
data_path = Path(__file__).parent / "data"
sys.path.insert(0, str(data_path))

from adaptable_agents import AdaptableOpenAIClient, ContextConfig  # noqa: E402
from load_dataset import load_locomo_dataset  # noqa: E402
from utils.locomo_metrics import calculate_metrics, aggregate_metrics  # noqa: E402
from utils.logger import setup_logging, logger  # noqa: E402

# System prompt for answering questions
SYSTEM_PROMPT = """You are an expert assistant that answers questions based on conversation context.

Your task is to answer questions accurately based on the provided context from past conversations.
- Use exact words from the context when possible
- For date questions, use approximate dates from the conversation
- For category 5 (adversarial) questions, select between the two provided options
- Keep answers short and concise
- If information is not in the context, say "Not mentioned in the conversation"

Answer format: Provide a short, direct answer based on the context."""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LoCoMo dataset evaluation with Adaptable Agents (KG strategy)"
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(Path(__file__).parent / "data" / "locomo10.json"),
        help="Path to the LoCoMo dataset JSON file (default: data/locomo10.json in the script directory)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--max_questions_per_sample",
        type=int,
        default=-1,
        help="Maximum number of questions per sample (-1 for all, default: -1)",
    )
    parser.add_argument(
        "--allow_categories",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Categories to evaluate (default: [1, 2, 3, 4, 5])",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-5.2",
        help="OpenAI model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens for generation (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--temperature_c5",
        type=float,
        default=0.5,
        help="Temperature for category 5 questions (default: 0.5)",
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
        default="locomo/kg",
        help="Memory scope path (default: locomo/kg)",
    )

    # Context configuration
    parser.add_argument(
        "--max_items",
        type=int,
        default=10,
        help="Maximum number of items in context (k parameter for AMEM, default: 10)",
    )
    parser.add_argument(
        "--evolve_top_k",
        type=int,
        default=1,
        help="Number of top memories to evolve (default: 1). Note: Currently hardcoded to 1 in the client, requires client modification to use this parameter.",
    )

    # Adaptable agents enable/disable
    parser.add_argument(
        "--enable_adaptable_agents",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to enable adaptable agents functionality (true/false, default: true)",
    )

    # Save path arguments
    parser.add_argument(
        "--save_directory",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )

    # Summarization configuration
    parser.add_argument(
        "--summarize_input",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Whether to summarize inputs before storage (true/false). If not provided, uses default configuration.",
    )

    return parser.parse_args()


def format_question_prompt(
    question: str,
    category: int,
    reference_answer: Optional[str] = None,
) -> str:
    """Format the question prompt based on category."""
    if category == 5:  # Adversarial question
        # For category 5, we need to provide two options
        if reference_answer:
            import random

            options = ["Not mentioned in the conversation", reference_answer]
            if random.random() < 0.5:
                options = options[::-1]  # Randomize order
            return f"""Question: {question}

Select the correct answer: {options[0]} or {options[1]}

Short answer:"""
        else:
            return f"""Question: {question}

Short answer:"""
    elif category == 2:  # Date question
        return f"""Answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question}

Short answer:"""
    elif category == 3:  # Inference question
        return f"""Write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question}

Short answer:"""
    else:  # Categories 1 and 4
        return f"""Write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question}

Short answer:"""


def load_conversations_into_memory(
    client: AdaptableOpenAIClient,
    samples: List,
    logger,
) -> None:
    """
    Load all conversation turns from samples into memory.

    This function calls load_prior_knowledge for each conversation turn
    to ensure they're loaded and indexed in the memory system.

    Args:
        client: The AdaptableOpenAIClient instance
        samples: List of LoCoMoSample objects
        logger: Logger instance for logging progress
    """
    logger.info("=" * 80)
    logger.info("Loading conversations into memory...")

    total_turns = 0
    loaded_count = 0

    # Count total turns first
    for sample in samples:
        for session_id, session in sample.conversation.sessions.items():
            total_turns += len(session.turns)

    logger.info(f"Total conversation turns to load: {total_turns}")

    # Call load_prior_knowledge for each turn
    for sample_idx, sample in enumerate(samples):
        for session_id, session in sample.conversation.sessions.items():
            # Skip empty sessions
            if not session.turns:
                continue

            logger.info(f"Loading conversation session {session_id} into memory ({len(session.turns)} turns)")

            # Load each turn individually
            for turn in session.turns:
                # Convert turn to a formatted string
                turn_text = f"[{session.date_time}] {turn.speaker}: {turn.text}"

                # Call load_prior_knowledge to load the turn into memory
                client.adaptable_agent.load_prior_knowledge(turn_text)
                loaded_count += 1

                # Log progress every 100 turns
                if loaded_count % 100 == 0:
                    logger.info(f"Loaded {loaded_count}/{total_turns} conversation turns into memory...")

    logger.info(f"Successfully loaded {loaded_count}/{total_turns} conversation turns into memory")
    logger.info("=" * 80)


def answer_question(
    client: AdaptableOpenAIClient,
    question: str,
    category: int,
    model_name: str,
    temperature: float,
    max_tokens: int,
    temperature_c5: float = 0.5,
    reference_answer: Optional[str] = None,
) -> str:
    """Answer a question using the adaptable agent client."""
    # Format the prompt based on category
    user_prompt = format_question_prompt(question, category, reference_answer)

    # Use appropriate temperature
    actual_temperature = temperature if category != 5 else temperature_c5

    # Make API call
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # temperature=actual_temperature,
            # max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""
        # Try to extract JSON if present
        if "answer" in answer.lower() and "{" in answer:
            try:
                import json

                # Try to find JSON in the response
                start = answer.find("{")
                end = answer.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = answer[start:end]
                    parsed = json.loads(json_str)
                    if "answer" in parsed:
                        return parsed["answer"]
            except Exception:
                pass

        return answer.strip()
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return ""


def main(args):
    """Main function to run the LoCoMo evaluation."""
    # Set up logging
    logs_dir = setup_logging()
    logger.info(f"LoCoMo evaluation logging initialized. Logs directory: {logs_dir}")
    logger.info("=" * 80)
    logger.info("Starting LoCoMo dataset evaluation (KG strategy)")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Model: {args.model_name}")
    logger.info("Strategy: kg (AMEM)")
    logger.info(f"Memory scope path: {args.memory_scope_path}")
    logger.info(f"Max items (k): {args.max_items}")
    logger.info(f"Evolve top k: {args.evolve_top_k}")
    logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
    logger.info(f"Allow categories: {args.allow_categories}")
    logger.info(f"Enable adaptable agents: {args.enable_adaptable_agents}")
    logger.info("=" * 80)

    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset_path}")
    if not Path(args.dataset_path).exists():
        logger.error(f"Dataset file not found: {args.dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")

    samples = load_locomo_dataset(args.dataset_path)
    logger.info(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        logger.info(f"Processing {len(samples)} samples (limited from total)")

    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
        )

    # Convert string flags to bool
    enable_adaptable_agents = args.enable_adaptable_agents.lower() == "true"
    summarize_input = None
    if args.summarize_input is not None:
        summarize_input = args.summarize_input.lower() == "true"

    # Initialize Adaptable OpenAI client
    logger.info("Initializing Adaptable OpenAI client...")
    context_config = ContextConfig(
        similarity_threshold=0.0,  # Not used for AMEM, but required
        max_items=args.max_items,
    )

    client = AdaptableOpenAIClient(
        adaptable_api_key=args.adaptable_api_key,
        openai_api_key=openai_api_key,
        api_base_url=args.api_base_url,
        memory_scope_path=args.memory_scope_path,
        context_config=context_config,
        auto_store_memories=True,
        summarize_input=summarize_input,
        strategy="kg",  # Use KG (AMEM) strategy
    )
    # Set enable_adaptable_agents property based on argument
    client.enable_adaptable_agents = enable_adaptable_agents
    logger.info("Adaptable OpenAI client initialized successfully")

    # Load all conversations into memory before processing questions
    if enable_adaptable_agents:
        load_conversations_into_memory(client, samples, logger)

    # Create save path
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_path = Path(args.save_directory) / "locomo" / f"kg_{timestamp}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save arguments
    args_path = save_path.with_suffix(".params.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize results
    results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    total_questions = 0

    # Process each sample
    logger.info("Starting to process samples...")
    for sample_idx, sample in enumerate(samples):
        logger.info("=" * 80)
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")
        logger.info(f"Sample ID: {sample.sample_id}")

        # Process questions
        questions_to_process = sample.qa
        if args.max_questions_per_sample > 0:
            questions_to_process = questions_to_process[: args.max_questions_per_sample]

        for qa_idx, qa in enumerate(questions_to_process):
            if qa.category not in args.allow_categories:
                continue

            total_questions += 1
            category_counts[qa.category] += 1

            logger.info("-" * 80)
            logger.info(
                f"Question {total_questions} (Sample {sample_idx}, QA {qa_idx})"
            )
            logger.info(f"Category: {qa.category}")
            logger.info(f"Question: {qa.question}")
            logger.info(f"Reference answer: {qa.final_answer}")

            # Answer the question (context is automatically fetched and appended by AdaptableOpenAIClient)
            try:
                prediction = answer_question(
                    client=client,
                    question=qa.question,
                    category=qa.category,
                    model_name=args.model_name,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    temperature_c5=args.temperature_c5,
                    reference_answer=qa.final_answer if qa.category == 5 else None,
                )
                logger.info(f"Category: {qa.category}")
                logger.info(f"Question: {qa.question}")
                logger.info(f"Reference answer: {qa.final_answer}")
                logger.info(f"Prediction: {prediction}")

                # Calculate metrics
                metrics = (
                    calculate_metrics(prediction, qa.final_answer)
                    if qa.final_answer
                    else {
                        "exact_match": 0,
                        "f1": 0.0,
                        "rouge1_f": 0.0,
                        "rouge2_f": 0.0,
                        "rougeL_f": 0.0,
                        "bleu1": 0.0,
                        "bleu2": 0.0,
                        "bleu3": 0.0,
                        "bleu4": 0.0,
                        "bert_f1": 0.0,
                        "meteor": 0.0,
                        "sbert_similarity": 0.0,
                    }
                )

                all_metrics.append(metrics)
                all_categories.append(qa.category)

                # Store result
                result = {
                    "sample_id": sample.sample_id,
                    "qa_idx": qa_idx,
                    "question": qa.question,
                    "prediction": prediction,
                    "reference": qa.final_answer,
                    "category": qa.category,
                    "metrics": metrics,
                }
                results.append(result)

                # Log metrics
                logger.info(f"Exact match: {metrics['exact_match']}")
                logger.info(f"F1: {metrics['f1']:.4f}")
                logger.info(f"ROUGE-L: {metrics['rougeL_f']:.4f}")
                logger.info(f"BERT F1: {metrics['bert_f1']:.4f}")

            except Exception as e:
                logger.error(f"Error processing question: {str(e)}", exc_info=True)
                results.append(
                    {
                        "sample_id": sample.sample_id,
                        "qa_idx": qa_idx,
                        "question": qa.question,
                        "error": str(e),
                        "category": qa.category,
                    }
                )

            # Save intermediate results
            if total_questions % 10 == 0:
                logger.info(
                    f"Processed {total_questions} questions, saving intermediate results..."
                )
                with open(save_path, "w") as f:
                    json.dump(
                        {
                            "results": results,
                            "total_questions": total_questions,
                            "category_counts": dict(category_counts),
                        },
                        f,
                        indent=2,
                    )

    # Calculate aggregate metrics
    logger.info("Calculating aggregate metrics...")
    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    # Prepare final results
    final_results = {
        "model": args.model_name,
        "strategy": "kg",
        "dataset": args.dataset_path,
        "total_questions": total_questions,
        "category_distribution": dict(category_counts),
        "aggregate_metrics": aggregate_results,
        "individual_results": results,
    }

    # Save final results
    logger.info(f"Saving final results to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info(f"Total questions: {total_questions}")
    logger.info(f"Category distribution: {dict(category_counts)}")
    logger.info("\nAggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        logger.info(f"\n{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict) and "mean" in stats:
                logger.info(
                    f"  {metric_name}: {stats['mean']:.4f} (std: {stats['std']:.4f})"
                )
    logger.info(f"\nResults saved to: {save_path}")
    logger.info(f"Logs saved to: {logs_dir}")
    logger.info("=" * 80)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print(f"Total questions: {total_questions}")
    print(f"Results saved to: {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
