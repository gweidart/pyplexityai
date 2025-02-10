"""OpenAI-compatible interface for Perplexity AI client."""

import time
from collections.abc import AsyncGenerator, Generator
from types import TracebackType
from typing import TypedDict, cast
from uuid import uuid4

from .async_client import AsyncPerplexityClient
from .client import PerplexityClient, SearchResponse
from .client_types import (
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ModelType,
)
from .errors import InvalidParameterError, PerplexityTimeoutError, SearchError


class PerplexityResult(TypedDict, total=False):
    """Type for Perplexity API result."""

    text: str
    finish_reason: str | None


def _convert_to_openai_response(
    result: SearchResponse,
    model: ModelType,
    messages: list[ChatMessage],
    stream: bool = False,
) -> ChatCompletionResponse | ChatCompletionStreamResponse:
    """
    Convert Perplexity AI response to OpenAI format.

    This function takes a Perplexity AI search response and converts it into a format
    compatible with OpenAI's chat completion responses. It can generate both streaming
    and non-streaming response formats.

    Args:
    ----
        result (SearchResponse): The search response from Perplexity AI.
        model (ModelType): The model used for the completion.
        messages (list[ChatMessage]): The list of chat messages in the conversation.
        stream (bool, optional): Whether to return a streaming response. Defaults to False.

    Returns:
    -------
        ChatCompletionResponse | ChatCompletionStreamResponse: The converted response
        in OpenAI format. If stream is True, returns a ChatCompletionStreamResponse,
        otherwise returns a ChatCompletionResponse.

    """
    response_id = str(uuid4())
    created = int(time.time())
    result_dict = cast(PerplexityResult, result)

    if stream:
        # For streaming responses
        return ChatCompletionStreamResponse(
            id=response_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatMessage(
                        role="assistant",
                        content=result_dict.get("text", ""),
                        name=None,
                    ),
                    finish_reason=result_dict.get("finish_reason"),
                )
            ],
        )

    # For non-streaming responses
    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[
            {
                "index": 0,
                "message": ChatMessage(
                    role="assistant",
                    content=result_dict.get("text", ""),
                    name=None,
                ),
                "finish_reason": result_dict.get("finish_reason", "stop"),
            }
        ],
        usage={
            "prompt_tokens": len(str(messages)) // 4,  # Rough estimation
            "completion_tokens": len(result_dict.get("text", "")) // 4,
            "total_tokens": (len(str(messages)) + len(result_dict.get("text", "")))
            // 4,
        },
    )


class OpenAICompatibleClient:
    """OpenAI-compatible client for Perplexity AI."""

    def __init__(
        self,
        api_key: str,
        *,
        api_version: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
        ----
            api_key: Your Perplexity AI API key
            api_version: Optional API version override
            max_retries: Maximum number of retries for failed requests
            timeout: Default timeout for requests in seconds

        Raises:
        ------
            AuthenticationError: If the API key is invalid or missing

        """
        self.sync_client = PerplexityClient(
            api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.async_client: AsyncPerplexityClient | None = None

    def create_chat_completion(
        self,
        messages: list[ChatMessage],
        *,
        model: ModelType = PerplexityClient.DEFAULT_MODEL,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool = False,
        stop: list[str] | str | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        user: str | None = None,
    ) -> Generator[ChatCompletionResponse | ChatCompletionStreamResponse, None, None]:
        """Create a chat completion (OpenAI-compatible).

        Args:
        ----
            messages: List of chat messages
            model: Model to use for completion
            temperature: Sampling temperature (not supported)
            top_p: Nucleus sampling parameter (not supported)
            n: Number of completions (not supported)
            stream: Whether to stream the response
            stop: Stop sequences (not supported)
            max_tokens: Maximum tokens to generate (not supported)
            presence_penalty: Presence penalty (not supported)
            frequency_penalty: Frequency penalty (not supported)
            logit_bias: Token biases (not supported)
            user: User identifier (not supported)

        Yields:
        ------
            Chat completion responses in OpenAI format

        Raises:
        ------
            InvalidParameterError: If parameters are invalid
            SearchError: If the completion request fails
            PerplexityTimeoutError: If the request times out

        """
        if not messages:
            raise InvalidParameterError("messages", "", ["non-empty list"])

        # Extract the last user message as the query
        last_user_msg = next(
            (msg for msg in reversed(messages) if msg["role"] == "user"),
            None,
        )
        if not last_user_msg:
            raise InvalidParameterError(
                "messages",
                str(messages),
                ["must contain at least one user message"],
            )

        query = last_user_msg["content"]
        mode = "copilot"  # Use copilot mode for chat

        try:
            for result in self.sync_client.search(
                query=query,
                mode=mode,
                model=model,
            ):
                # Always use stream=True for consistency since we're in a generator
                yield _convert_to_openai_response(result, model, messages, True)
        except Exception as e:
            if isinstance(
                e,
                InvalidParameterError | SearchError | PerplexityTimeoutError,
            ):
                raise
            raise SearchError(query, mode, e) from e

    async def acreate_chat_completion(
        self,
        messages: list[ChatMessage],
        *,
        model: ModelType = PerplexityClient.DEFAULT_MODEL,
        temperature: float | None = None,
        top_p: float | None = None,
        n: int | None = None,
        stream: bool = False,
        stop: list[str] | str | None = None,
        max_tokens: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        user: str | None = None,
    ) -> AsyncGenerator[ChatCompletionResponse | ChatCompletionStreamResponse, None]:
        """Create a chat completion asynchronously (OpenAI-compatible).

        Args:
        ----
            messages: List of chat messages
            model: Model to use for completion
            temperature: Sampling temperature (not supported)
            top_p: Nucleus sampling parameter (not supported)
            n: Number of completions (not supported)
            stream: Whether to stream the response
            stop: Stop sequences (not supported)
            max_tokens: Maximum tokens to generate (not supported)
            presence_penalty: Presence penalty (not supported)
            frequency_penalty: Frequency penalty (not supported)
            logit_bias: Token biases (not supported)
            user: User identifier (not supported)

        Yields:
        ------
            Chat completion responses in OpenAI format

        Raises:
        ------
            InvalidParameterError: If parameters are invalid
            SearchError: If the completion request fails
            PerplexityTimeoutError: If the request times out

        """
        if not messages:
            raise InvalidParameterError("messages", "", ["non-empty list"])

        # Extract the last user message as the query
        last_user_msg = next(
            (msg for msg in reversed(messages) if msg["role"] == "user"),
            None,
        )
        if not last_user_msg:
            raise InvalidParameterError(
                "messages",
                str(messages),
                ["must contain at least one user message"],
            )

        query = last_user_msg["content"]
        mode = "copilot"  # Use copilot mode for chat

        # Create async client if needed
        if self.async_client is None:
            self.async_client = AsyncPerplexityClient(
                self.sync_client.api_key,
                api_version=self.sync_client.api_version,
                max_retries=self.sync_client.max_retries,
                timeout=self.sync_client.default_timeout,
            )

        try:
            async for result in self.async_client.search(
                query=query,
                mode=mode,
                model=model,
            ):
                yield _convert_to_openai_response(result, model, messages, stream)
        except Exception as e:
            if isinstance(
                e,
                InvalidParameterError | SearchError | PerplexityTimeoutError,
            ):
                raise
            raise SearchError(query, mode, e) from e

    async def close(self) -> None:
        """
        Close the client asynchronously (deprecated).

        This method is deprecated. Use `aclose()` instead.

        This method closes any open connections and performs cleanup operations.
        It delegates the actual closing process to the `aclose()` method.

        Returns
        -------
            None

        """
        await self.aclose()

    async def aclose(self) -> None:
        """
        Asynchronously close the client and release resources.

        This method closes both the asynchronous and synchronous clients,
        releasing any resources they might be holding. It ensures that
        all connections are properly closed and cleaned up.

        If an asynchronous client exists, it is closed first using its
        async_close method. Then, the synchronous client is closed.
        Finally, the reference to the asynchronous client is removed.

        Returns
        -------
            None

        """
        if self.async_client:
            await self.async_client.async_close()
            self.async_client = None
        self.sync_client.close()

    async def __aenter__(self) -> "OpenAICompatibleClient":
        """
        Asynchronous context manager entry point.

        This method is called when entering an asynchronous context using the 'async with' statement.
        It allows the OpenAICompatibleClient to set up any necessary resources or perform
        initialization tasks before the context block is executed.

        Returns
        -------
        OpenAICompatibleClient
            Returns the instance of the OpenAICompatibleClient, allowing it to be used
            within the context manager block.

        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Asynchronous context manager exit point.

        This method is called when exiting an asynchronous context using the 'async with' statement.
        It ensures proper cleanup of resources by calling the aclose() method.

        Args:
        ----
            exc_type (type[BaseException] | None): The type of the exception that caused the context to be exited,
                                                   or None if no exception was raised.
            exc_val (BaseException | None): The instance of the exception that caused the context to be exited,
                                            or None if no exception was raised.
            exc_tb (TracebackType | None): A traceback object encapsulating the call stack at the point where
                                           the exception originally occurred, or None if no exception was raised.

        Returns:
        -------
            None

        """
        await self.aclose()
