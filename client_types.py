"""Type definitions for Perplexity AI client."""

from typing import Literal, TypedDict

# Model types
ModelType = Literal[
    "sonar-pro",
    "sonar-reasoning-pro",
    "sonar-reasoning",
    "sonar",
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
    # Legacy models
    "llama-3-sonar-small-32k-online",
    "llama-3-sonar-large-32k-online",
    "llama-3-sonar-small-32k-chat",
    "llama-3-sonar-large-32k-chat",
    "llama-3-8b-instruct",
    "llama-3-70b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct",
]

# Search modes
SearchMode = Literal["concise", "copilot"]

# Search focus types
SearchFocus = Literal[
    "internet",
    "scholar",
    "writing",
    "wolfram",
    "youtube",
    "reddit",
]


# OpenAI-compatible types
class ChatMessage(TypedDict):
    """OpenAI-compatible chat message."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: str | None


class ChatCompletionRequest(TypedDict, total=False):
    """OpenAI-compatible chat completion request."""

    model: ModelType
    messages: list[ChatMessage]
    temperature: float | None
    top_p: float | None
    n: int | None
    stream: bool | None
    stop: list[str] | str | None
    max_tokens: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    logit_bias: dict[str, float] | None
    user: str | None


class ChatCompletionChoice(TypedDict):
    """OpenAI-compatible chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionUsage(TypedDict):
    """OpenAI-compatible token usage stats."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(TypedDict):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(TypedDict):
    """OpenAI-compatible streaming chat completion choice."""

    index: int
    delta: ChatMessage
    finish_reason: str | None


class ChatCompletionStreamResponse(TypedDict):
    """OpenAI-compatible streaming chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]
