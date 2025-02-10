"""A clean, simple Perplexity AI API client."""

from .async_client import AsyncPerplexityClient
from .client import PerplexityClient
from .client_types import ChatMessage, ModelType
from .errors import (
    AuthenticationError,
    InvalidParameterError,
    PerplexityTimeoutError,
    SearchError,
)
from .openai import OpenAICompatibleClient

__version__ = "0.1.0"
__all__ = [
    "AsyncPerplexityClient",
    "AuthenticationError",
    "ChatMessage",
    "InvalidParameterError",
    "ModelType",
    "OpenAICompatibleClient",
    "PerplexityClient",
    "PerplexityTimeoutError",
    "SearchError",
]
