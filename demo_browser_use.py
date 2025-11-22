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
from browser_use.llm.messages import BaseMessage
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.openai.serializer import OpenAIMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from openai import AsyncOpenAI, OpenAI
from openai.types.shared_params.response_format_json_schema import (
    JSONSchema,
    ResponseFormatJSONSchema,
)

load_dotenv()

# Try to import browser-use LLM classes


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
        self.model_name = model
        self.provider = "openai"
        # Create AsyncOpenAI client for direct calls (when images are present)
        self.async_openai_client = AsyncOpenAI(
            api_key=adaptable_client.openai_client.api_key
        )

    @property
    def name(self) -> str:
        return str(self.model)

    def _has_images(self, messages: list[dict]) -> bool:
        """Check if messages contain any images."""
        for message in messages:
            content = message.get("content", "")
            # Check if content is a list (multimodal content)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        return True
        return False

    def _get_usage(self, response) -> ChatInvokeUsage | None:
        """Extract usage information from response, matching browser-use pattern."""
        if response.usage is not None:
            completion_tokens = response.usage.completion_tokens
            completion_token_details = response.usage.completion_tokens_details
            if completion_token_details is not None:
                reasoning_tokens = completion_token_details.reasoning_tokens
                if reasoning_tokens is not None:
                    completion_tokens += reasoning_tokens

            return ChatInvokeUsage(
                prompt_tokens=response.usage.prompt_tokens,
                prompt_cached_tokens=response.usage.prompt_tokens_details.cached_tokens
                if response.usage.prompt_tokens_details is not None
                else None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return None

    async def ainvoke(
        self, messages: list[BaseMessage], output_format: type | None = None
    ) -> ChatInvokeCompletion:
        """Async invoke method expected by browser-use Agent."""
        # Convert BaseMessage objects to OpenAI format
        openai_messages = OpenAIMessageSerializer.serialize_messages(messages)

        # Check if messages contain images
        has_images = self._has_images(openai_messages)

        # Prepare model parameters
        model_params: dict[str, Any] = {}

        # If images are present, use AsyncOpenAI directly (bypass adaptable agent)
        # Otherwise, use AdaptableOpenAIClient for context-aware responses
        if has_images:
            # Use async client directly for image requests
            if output_format is None:
                response = await self.async_openai_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    **model_params,
                )
                usage = self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=response.choices[0].message.content or "",
                    usage=usage,
                    stop_reason=response.choices[0].finish_reason
                    if response.choices
                    else None,
                )
            else:
                # Handle structured output for images
                response_format: JSONSchema = {
                    "name": "agent_output",
                    "strict": True,
                    "schema": SchemaOptimizer.create_optimized_json_schema(
                        output_format
                    ),
                }
                response = await self.async_openai_client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    response_format=ResponseFormatJSONSchema(
                        json_schema=response_format, type="json_schema"
                    ),
                    **model_params,
                )
                if response.choices[0].message.content is None:
                    raise ValueError(
                        "Failed to parse structured output from model response"
                    )
                usage = self._get_usage(response)
                parsed = output_format.model_validate_json(
                    response.choices[0].message.content
                )
                return ChatInvokeCompletion(
                    completion=parsed,
                    usage=usage,
                    stop_reason=response.choices[0].finish_reason
                    if response.choices
                    else None,
                )
        else:
            # Use AdaptableOpenAIClient for text-only requests
            # Run synchronous calls in a thread to avoid blocking
            if output_format is None:
                response = await asyncio.to_thread(
                    self.adaptable_client.chat.completions.create,
                    model=self.model,
                    messages=openai_messages,
                )
                usage = self._get_usage(response)
                return ChatInvokeCompletion(
                    completion=response.choices[0].message.content or "",
                    usage=usage,
                    stop_reason=response.choices[0].finish_reason
                    if response.choices
                    else None,
                )
            else:
                # For structured output with adaptable agent, we need to request JSON format
                # Note: AdaptableOpenAIClient might not support response_format directly
                # So we'll get the response and parse it
                response = await asyncio.to_thread(
                    self.adaptable_client.chat.completions.create,
                    model=self.model,
                    messages=openai_messages,
                )
                if response.choices[0].message.content is None:
                    raise ValueError(
                        "Failed to parse structured output from model response"
                    )
                usage = self._get_usage(response)
                # Parse the JSON response
                parsed = output_format.model_validate_json(
                    response.choices[0].message.content
                )
                return ChatInvokeCompletion(
                    completion=parsed,
                    usage=usage,
                    stop_reason=response.choices[0].finish_reason
                    if response.choices
                    else None,
                )


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

        # Initialize browser with timeout
        print("Initializing browser... (this may take 30-60 seconds on first run)")
        browser = Browser()
        try:
            # Add a longer timeout for browser startup (90 seconds to account for extension downloads)
            await asyncio.wait_for(browser.start(), timeout=90.0)
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Browser startup timed out after 90 seconds. "
                "This may happen if:\n"
                "  1. Chromium is not installed - run: uvx browser-use install\n"
                "  2. Extensions are downloading (first run only)\n"
                "  3. System resources are limited\n"
                "Try running the demo again, or check browser-use documentation."
            )

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
            await browser.stop()

    except TimeoutError as e:
        print(f"\n❌ Browser Timeout Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Install Chromium: uvx browser-use install")
        print("2. Check if another browser process is running")
        print("3. Try running the demo again")
        response_text = str(e)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")

        # Provide helpful error messages for common issues
        if "browser" in error_msg.lower() or "chromium" in error_msg.lower():
            print("\nTroubleshooting tips:")
            print("1. Install Chromium: uvx browser-use install")
            print("2. Check browser-use documentation: https://docs.browser-use.com/")

        # Only print full traceback for unexpected errors
        if "TimeoutError" not in error_msg and "browser" not in error_msg.lower():
            import traceback

            traceback.print_exc()

        response_text = error_msg

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

        # Initialize browser with timeout
        print("Initializing browser... (this may take 30-60 seconds on first run)")
        browser = Browser()
        try:
            # Add a longer timeout for browser startup (90 seconds to account for extension downloads)
            await asyncio.wait_for(browser.start(), timeout=90.0)
        except asyncio.TimeoutError:
            raise TimeoutError(
                "Browser startup timed out after 90 seconds. "
                "This may happen if:\n"
                "  1. Chromium is not installed - run: uvx browser-use install\n"
                "  2. Extensions are downloading (first run only)\n"
                "  3. System resources are limited\n"
                "Try running the demo again, or check browser-use documentation."
            )

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
            await browser.stop()

    except TimeoutError as e:
        print(f"\n❌ Browser Timeout Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Install Chromium: uvx browser-use install")
        print("2. Check if another browser process is running")
        print("3. Try running the demo again")
        response_text = str(e)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")

        # Provide helpful error messages for common issues
        if "browser" in error_msg.lower() or "chromium" in error_msg.lower():
            print("\nTroubleshooting tips:")
            print("1. Install Chromium: uvx browser-use install")
            print("2. Check browser-use documentation: https://docs.browser-use.com/")

        # Only print full traceback for unexpected errors
        if "TimeoutError" not in error_msg and "browser" not in error_msg.lower():
            import traceback

            traceback.print_exc()

        response_text = error_msg

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

    # Check if browser is likely installed (basic check)
    print("Note: Make sure Chromium is installed. If not, run: uvx browser-use install")
    print(
        "Browser initialization may take longer on first run as extensions are downloaded.\n"
    )

    adaptable_api_key = os.getenv("ADAPTABLE_API_KEY")
    if not adaptable_api_key:
        print(
            "WARNING: ADAPTABLE_API_KEY not found. Running only standard OpenAI demo."
        )
        print(
            "Set ADAPTABLE_API_KEY in your .env file to enable adaptable agents comparison."
        )

    # Run with standard OpenAI
    # standard_results = await run_with_standard_openai(task)

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

    standard_results = None  # Can be set by uncommenting the line above
    if standard_results and adaptable_results:
        print(f"\n{'Metric':<30} {'Standard OpenAI':<20} {'Adaptable Agent':<20}")
        print("-" * 70)
        print(
            f"{'Success':<30} {str(standard_results['success']):<20} {str(adaptable_results['success']):<20}"
        )
        adaptable_duration = f"{adaptable_results['duration']:.2f}"
        print(
            f"{'Duration (seconds)':<30} {standard_results['duration']:<20.2f} {adaptable_duration:<20}"
        )
        print(
            f"{'Tokens (estimated)':<30} {standard_results['tokens_estimated']:<20} {adaptable_results['tokens_estimated']:<20}"
        )

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
        print(f"\nAdaptable Agent Response:\n{adaptable_results['response']}")
    elif adaptable_results:
        print(f"\n{'Metric':<30} {'Adaptable Agent':<20}")
        print("-" * 50)
        print(f"{'Success':<30} {str(adaptable_results['success']):<20}")
        print(f"{'Duration (seconds)':<30} {adaptable_results['duration']:<20.2f}")
        print(f"{'Tokens (estimated)':<30} {adaptable_results['tokens_estimated']:<20}")
        print("\n" + "=" * 60)
        print("Response")
        print("=" * 60)
        print(f"\nAdaptable Agent Response:\n{adaptable_results['response']}")

    print("\n" + "=" * 60)
    print(
        "Note: Adaptable Agents learn from each run, so performance improves over time!"
    )
    print("Run this demo multiple times to see the improvement.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
