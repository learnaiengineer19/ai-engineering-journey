
"""
LESSON 18 - PART 8: MODEL FACTORY
================================

PURPOSE:
Create different LangChain chat models using
one shared interface.

SUPPORTED:
- ollama
- openai
- bedrock

The default provider is local Ollama.
"""

import os

from langchain_core.language_models.chat_models import (
    BaseChatModel
)


def get_model(
    provider: str | None = None,
    model_name: str | None = None
) -> BaseChatModel:
    """
    Build a LangChain chat model based on the
    selected provider.

    Provider credentials are read from the
    environment, not hardcoded in Python.
    """

    provider = (
        provider
        or os.getenv("LLM_PROVIDER", "ollama")
    ).lower()

    if provider == "ollama":

        from langchain_ollama import ChatOllama

        name = (
            model_name
            or os.getenv("OLLAMA_MODEL", "qwen2:7b")
        )

        return ChatOllama(
            model=name,
            temperature=0
        )

    elif provider == "openai":

        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is required"
            )

        name = (
            model_name
            or os.getenv("OPENAI_MODEL")
        )

        if not name:
            raise ValueError(
                "Set OPENAI_MODEL to a supported model ID"
            )

        return ChatOpenAI(
            model=name
        )

    elif provider == "bedrock":

        from langchain_aws import ChatBedrockConverse

        name = (
            model_name
            or os.getenv("BEDROCK_MODEL_ID")
        )

        region = (
            os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
        )

        if not name or not region:
            raise ValueError(
                "BEDROCK_MODEL_ID and AWS_REGION "
                "are required"
            )

        # Boto3 / AWS SDK resolves credentials
        # through its normal credential chain.
        return ChatBedrockConverse(
            model_id=name,
            region_name=region
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )
