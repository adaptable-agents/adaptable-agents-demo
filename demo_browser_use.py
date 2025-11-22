"""
Demo comparing browser-use agent performance with and without Adaptable Agents.

This demo runs the same browser automation task using:
1. Standard OpenAI client (without adaptable agents)
2. AdaptableOpenAIClient (with adaptable agents)

The task: "Find the number of stars of the browser-use repo on GitHub"
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add adaptable-agents-python-package to path
sys.path.insert(
    0, str(Path(__file__).parent.parent / "adaptable-agents-python-package")
)

from adaptable_agents import AdaptableOpenAIClient, ContextConfig
from browser_use import Agent, Browser
from openai import OpenAI

load_dotenv()

# Try to import browser-use LLM classes
try:
    from browser_use.llm.openai import ChatOpenAI
except ImportError:
    # Fallback if browser-use structure is different
    ChatOpenAI = None


class AdaptableBrowserLLM:
    """
    Custom LLM wrapper for browser-use that uses AdaptableOpenAIClient.

    This class implements the interface expected by browser-use Agent.
    """

    def __init__(
        self, adaptable_client: AdaptableOpenAIClient, model: str = "gpt-4o-mini"
    ):
        self.adaptable_client = adaptable_client
        self.model = model

    async def ainvoke(self, messages: list[dict], **kwargs):
        """Async invoke method expected by browser-use Agent."""
        response = self.adaptable_client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response.choices[0].message.content or ""


async def run_with_standard_openai(
    task: str, model: str = "gpt-4o-mini"
) -> dict[str, Any]:
    """
    Run browser automation task using standard OpenAI client.

    Args:
        task: The task description
        model: OpenAI model to use

    Returns:
        Dictionary with metrics: success, duration, tokens, response
    """
    print("\n" + "=" * 60)
    print("Running with STANDARD OpenAI Client (No Adaptable Agents)")
    print("=" * 60)

    start_time = time.time()
    success = False
    response_text = ""
    tokens_used = 0

    try:
        # Use browser-use's ChatOpenAI if available, otherwise create custom wrapper
        if ChatOpenAI:
            llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model)
        else:
            # Fallback: create a simple wrapper
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            class StandardLLM:
                def __init__(self, client: OpenAI, model: str):
                    self.client = client
                    self.model = model

                async def ainvoke(self, messages: list[dict], **kwargs):
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages, **kwargs
                    )
                    return response.choices[0].message.content or ""

            llm = StandardLLM(openai_client, model)

        # Initialize browser
        browser = Browser()
        await browser.start()

        try:
            # Create agent with standard LLM
            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
            )

            # Run the agent
            history = await agent.run()

            # Extract response from history
            if history and len(history) > 0:
                last_message = history[-1]
                if isinstance(last_message, dict):
                    response_text = last_message.get("content", str(last_message))
                else:
                    response_text = str(last_message)

            success = True

        finally:
            await browser.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        response_text = str(e)

    duration = time.time() - start_time

    # Try to estimate tokens (rough approximation)
    if response_text:
        tokens_used = len(response_text.split()) * 1.3  # Rough estimate

    return {
        "success": success,
        "duration": duration,
        "tokens_estimated": int(tokens_used),
        "response": response_text[:200] + "..."
        if len(response_text) > 200
        else response_text,
    }


async def run_with_adaptable_agent(
    task: str,
    model: str = "gpt-4o-mini",
    memory_scope_path: str = "browser-use/demo",
    similarity_threshold: float = 0.8,
    max_items: int = 5,
) -> dict[str, Any]:
    """
    Run browser automation task using AdaptableOpenAIClient.

    Args:
        task: The task description
        model: OpenAI model to use
        memory_scope_path: Memory scope for adaptable agents
        similarity_threshold: Similarity threshold for context retrieval
        max_items: Maximum number of past experiences to use

    Returns:
        Dictionary with metrics: success, duration, tokens, response
    """
    print("\n" + "=" * 60)
    print("Running with ADAPTABLE OpenAI Client (With Adaptable Agents)")
    print("=" * 60)

    start_time = time.time()
    success = False
    response_text = ""
    tokens_used = 0

    try:
        # Initialize AdaptableOpenAIClient
        adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
        api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

        if not adaptable_api_key:
            raise ValueError("ADAPTABLE_API_KEY not found in environment")

        context_config = ContextConfig(
            similarity_threshold=similarity_threshold,
            max_items=max_items,
        )

        adaptable_client = AdaptableOpenAIClient(
            adaptable_api_key=adaptable_api_key,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            api_base_url=api_base_url,
            memory_scope_path=memory_scope_path,
            context_config=context_config,
            auto_store_memories=True,
            enable_adaptable_agents=True,
        )

        # Create browser-use compatible LLM wrapper
        llm = AdaptableBrowserLLM(adaptable_client, model)

        # Initialize browser
        browser = Browser()
        await browser.start()

        try:
            # Create agent with adaptable LLM
            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
            )

            # Run the agent
            history = await agent.run()

            # Extract response from history
            if history and len(history) > 0:
                last_message = history[-1]
                if isinstance(last_message, dict):
                    response_text = last_message.get("content", str(last_message))
                else:
                    response_text = str(last_message)

            success = True

        finally:
            await browser.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        response_text = str(e)

    duration = time.time() - start_time

    # Try to estimate tokens (rough approximation)
    if response_text:
        tokens_used = len(response_text.split()) * 1.3  # Rough estimate

    return {
        "success": success,
        "duration": duration,
        "tokens_estimated": int(tokens_used),
        "response": response_text[:200] + "..."
        if len(response_text) > 200
        else response_text,
    }


async def main():
    """Run comparison demo."""
    print("\n" + "=" * 60)
    print("Browser-Use Performance Comparison: Standard vs Adaptable Agent")
    print("=" * 60)

    # Task: Find the number of stars of the browser-use repo
    task = "Find the number of stars of the browser-use repo on GitHub (https://github.com/browser-use/browser-use)"

    print(f"\nTask: {task}\n")

    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file or as an environment variable")
        return

    adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
    if not adaptable_api_key:
        print(
            "WARNING: ADAPTABLE_API_KEY not found. Running only standard OpenAI demo."
        )
        print(
            "Set ADAPTABLE_API_KEY in your .env file to enable adaptable agents comparison."
        )

    # Run with standard OpenAI
    standard_results = await run_with_standard_openai(task)

    # Run with adaptable agent (if API key is available)
    adaptable_results = None
    if adaptable_api_key:
        adaptable_results = await run_with_adaptable_agent(
            task,
            memory_scope_path="browser-use/github-stars-demo",
        )

    # Print comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'Standard OpenAI':<20} {'Adaptable Agent':<20}")
    print("-" * 70)
    print(
        f"{'Success':<30} {str(standard_results['success']):<20} {str(adaptable_results['success']) if adaptable_results else 'N/A':<20}"
    )
    print(
        f"{'Duration (seconds)':<30} {standard_results['duration']:.2f:<20} {adaptable_results['duration']:.2f if adaptable_results else 'N/A':<20}"
    )
    print(
        f"{'Tokens (estimated)':<30} {standard_results['tokens_estimated']:<20} {adaptable_results['tokens_estimated'] if adaptable_results else 'N/A':<20}"
    )

    if adaptable_results:
        improvement = (
            (standard_results["duration"] - adaptable_results["duration"])
            / standard_results["duration"]
        ) * 100
        print(f"\n{'Time Improvement':<30} {improvement:+.1f}%")

        if adaptable_results["duration"] < standard_results["duration"]:
            print("\n✅ Adaptable Agent completed the task FASTER!")
        elif adaptable_results["duration"] > standard_results["duration"]:
            print(
                "\n⚠️  Standard OpenAI was faster (but adaptable agent may improve with more runs)"
            )
        else:
            print("\n➡️  Similar performance")

    print("\n" + "=" * 60)
    print("Response Samples")
    print("=" * 60)
    print(f"\nStandard OpenAI Response:\n{standard_results['response']}")
    if adaptable_results:
        print(f"\nAdaptable Agent Response:\n{adaptable_results['response']}")

    print("\n" + "=" * 60)
    print(
        "Note: Adaptable Agents learn from each run, so performance improves over time!"
    )
    print("Run this demo multiple times to see the improvement.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
