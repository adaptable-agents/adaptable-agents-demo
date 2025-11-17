"""
AdaptableAgents model wrapper for mini-swe-agent integration.
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptable_agents import AdaptableOpenAIClient, ContextConfig

logger = logging.getLogger("adaptable_model")


@dataclass
class AdaptableModelConfig:
    """Configuration for AdaptableAgents model."""

    model_name: str
    adaptable_api_key: str
    openai_api_key: str | None = None
    api_base_url: str = "http://localhost:8000"
    memory_scope_path: str = "default"
    similarity_threshold: float = 0.8
    max_items: int = 5
    auto_store_memories: bool = True
    summarize_input: bool | None = None
    enable_adaptable_agents: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class AdaptableModel:
    """
    Model wrapper that uses AdaptableOpenAIClient for mini-swe-agent integration.

    This class implements the Model protocol required by mini-swe-agent and wraps
    the AdaptableOpenAIClient to provide context-aware LLM calls.
    """

    def __init__(self, *, config_class: type = AdaptableModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0

        # Get OpenAI API key from environment if not provided
        openai_api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
            )

        # Create context config
        context_config = ContextConfig(
            similarity_threshold=self.config.similarity_threshold,
            max_items=self.config.max_items,
        )

        # Initialize AdaptableOpenAIClient
        self.client = AdaptableOpenAIClient(
            adaptable_api_key=self.config.adaptable_api_key,
            openai_api_key=openai_api_key,
            api_base_url=self.config.api_base_url,
            memory_scope_path=self.config.memory_scope_path,
            context_config=context_config,
            auto_store_memories=self.config.auto_store_memories,
            summarize_input=self.config.summarize_input,
            enable_adaptable_agents=self.config.enable_adaptable_agents,
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """
        Query the model using AdaptableOpenAIClient.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the API

        Returns:
            Dictionary with 'content' and 'extra' keys matching the Model protocol
        """
        # Merge model_kwargs with kwargs
        api_kwargs = self.config.model_kwargs | kwargs

        # Call the AdaptableOpenAIClient
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            **api_kwargs,
        )

        # Extract content
        content = response.choices[0].message.content or ""

        # Calculate cost (simplified - you may want to use actual cost calculation)
        # For now, we'll track calls but not calculate precise costs
        self.n_calls += 1

        # Try to extract usage information if available
        if hasattr(response, "usage") and response.usage:
            # You can add cost calculation here based on usage
            pass

        return {
            "content": content,
            "extra": {
                "response": response.model_dump()
                if hasattr(response, "model_dump")
                else str(response),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        """Get template variables for rendering prompts."""
        return asdict(self.config) | {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
        }
