"""Perplexity AI API client implementation."""

import abc
import json
import logging
from collections.abc import AsyncGenerator, Generator, Sequence
from datetime import datetime, timezone
from time import time
from types import TracebackType
from typing import Any, TypeAlias
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter, Retry

from .client_types import ModelType, SearchFocus, SearchMode
from .errors import (
    AuthenticationError,
    InvalidParameterError,
    PerplexityTimeoutError,
    SearchError,
)

# Configure logging
logger = logging.getLogger("perplexity_client")

# Type aliases for response types
SearchResponse: TypeAlias = dict[str, Any]


class BasePerplexityClient(abc.ABC):
    """Abstract base class for Perplexity AI clients."""

    API_BASE = "https://api.perplexity.ai"
    API_VERSION = "2024-03-15"  # Latest stable API version
    CLIENT_VERSION = "0.1.0"

    # API Endpoints
    CHAT_ENDPOINT = "/chat/completions"

    VALID_MODES: Sequence[SearchMode] = ["concise", "copilot"]
    VALID_SEARCH_FOCUS: Sequence[SearchFocus] = [
        "internet",
        "scholar",
        "writing",
        "wolfram",
        "youtube",
        "reddit",
    ]

    VALID_MODELS: Sequence[ModelType] = [
        # Current generation models (recommended)
        "sonar-pro",
        "sonar-reasoning-pro",
        "sonar-reasoning",
        "sonar",
        "llama-3.1-70b-instruct",
        "llama-3.1-8b-instruct",
        # Legacy models (will be deprecated August 12, 2024)
        "llama-3-sonar-small-32k-online",
        "llama-3-sonar-large-32k-online",
        "llama-3-sonar-small-32k-chat",
        "llama-3-sonar-large-32k-chat",
        "llama-3-8b-instruct",
        "llama-3-70b-instruct",
        "mistral-7b-instruct",
        "mixtral-8x7b-instruct",
    ]

    # Model context windows
    MODEL_CONTEXT_WINDOWS = {  # noqa: RUF012
        "llama-3.1-sonar-small-128k-online": 128_000,
        "llama-3.1-sonar-large-128k-online": 128_000,
        "llama-3.1-sonar-small-128k-chat": 128_000,
        "llama-3.1-sonar-large-128k-chat": 128_000,
        "llama-3.1-70b-instruct": 4096,
        "llama-3.1-8b-instruct": 4096,
        "llama-3-sonar-small-32k-online": 32_000,
        "llama-3-sonar-large-32k-online": 32_000,
        "llama-3-sonar-small-32k-chat": 32_000,
        "llama-3-sonar-large-32k-chat": 32_000,
        "llama-3-8b-instruct": 4096,
        "llama-3-70b-instruct": 4096,
        "mistral-7b-instruct": 16_384,
        "mixtral-8x7b-instruct": 16_384,
    }

    # Default to latest recommended model
    DEFAULT_MODEL: ModelType = "sonar-pro"

    def __init__(
        self,
        api_key: str,
        *,
        api_version: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        logger: logging.Logger | None = None,
        default_model: str = "sonar-pro",
        base_url: str = "https://api.perplexity.ai",
    ) -> None:
        """
        Initialize the Perplexity AI client with the provided configuration.

        Args:
        ----
            api_key (str): The API key for authentication with the Perplexity AI service.
            api_version (str | None, optional): The API version to use. If None, the default version will be used. Defaults to None.
            max_retries (int, optional): The maximum number of retries for failed requests. Defaults to 3.
            timeout (float, optional): The default timeout for API requests in seconds. Defaults to 30.0.
            logger (logging.Logger | None, optional): A custom logger instance. If None, a default logger will be used. Defaults to None.
            default_model (str, optional): The default model to use for API requests. Defaults to "sonar-pro".
            base_url (str, optional): The base URL for the Perplexity AI API. Defaults to "https://api.perplexity.ai".

        Raises:
        ------
            AuthenticationError: If the provided API key is empty or invalid.

        Returns:
        -------
            None

        """
        if not api_key:
            raise AuthenticationError(api_key)

        self.api_key = api_key
        self.api_version = api_version or self.API_VERSION
        self.max_retries = max_retries
        self.default_timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self.default_model = default_model
        self.base_url = base_url

        # Request tracking
        self._request_id = str(uuid4())
        self._request_start = None
        self._last_request_latency = None
        self._total_requests = 0
        self._failed_requests = 0

    def _log_request_start(self, operation: str) -> None:
        """
        Log the start of a request operation.

        This method records the start time of a request, increments the total request count,
        and logs debug information about the starting request.

        Args:
        ----
            operation (str): The name or type of the operation being performed.

        Returns:
        -------
            None

        Side effects:
        ------------
            - Sets the _request_start time
            - Increments _total_requests
            - Logs debug information using the logger

        """
        self._request_start = time()
        self._total_requests += 1
        self.logger.debug(
            "Starting %s request (ID: %s)",
            operation,
            self._request_id,
            extra={
                "operation": operation,
                "request_id": self._request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _log_request_success(self, operation: str) -> None:
        """
        Log a successful request with details about the operation and latency.

        This method calculates the request latency, updates the last request latency,
        and logs debug information about the successful request.

        Args:
        ----
            operation (str): The name or type of the operation that was performed successfully.

        Returns:
        -------
            None

        Side effects:
        ------------
            - Updates self._last_request_latency with the calculated latency.
            - Logs debug information using the logger, including operation details,
              request ID, latency, and timestamp.

        """
        if self._request_start:
            self._last_request_latency = time() - self._request_start
            self.logger.debug(
                "%s request completed successfully in %.2fs (ID: %s)",
                operation,
                self._last_request_latency,
                self._request_id,
                extra={
                    "operation": operation,
                    "request_id": self._request_id,
                    "latency": self._last_request_latency,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _log_request_failure(self, operation: str, error: Exception) -> None:
        """
        Log details of a failed request.

        This method increments the failed request counter, calculates the request latency,
        and logs an error message with detailed information about the failed request.

        Args:
        ----
            operation (str): The name or type of the operation that failed.
            error (Exception): The exception that caused the request to fail.

        Returns:
        -------
            None

        Side effects:
        ------------
            - Increments the _failed_requests counter.
            - Logs an error message with request details using the logger.

        """
        self._failed_requests += 1
        if self._request_start:
            latency = time() - self._request_start
            self.logger.error(
                "%s request failed after %.2fs: %s (ID: %s)",
                operation,
                latency,
                str(error),
                self._request_id,
                extra={
                    "operation": operation,
                    "request_id": self._request_id,
                    "latency": latency,
                    "error": str(error),
                    "error_type": error.__class__.__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                exc_info=True,
            )

    def _log_retry(self, retry_state: Retry) -> None:
        """
        Log information about retry attempts for failed requests.

        This method logs a warning message with details about the current retry attempt,
        including the attempt number and the maximum number of retries allowed.

        Args:
        ----
            retry_state (Retry): An object containing information about the current
                                 retry state, including the number of observed errors
                                 and the total number of allowed retries.

        Returns:
        -------
            None

        Side effects:
        ------------
            Logs a warning message using the logger with retry attempt information.

        """
        self.logger.warning(
            "Retrying request (attempt %d/%d)",
            retry_state._observed_errors,  # type: ignore
            retry_state.total,
            extra={
                "request_id": self._request_id,
                "retry_number": retry_state._observed_errors,  # type: ignore
                "max_retries": retry_state.total,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _validate_model_context(self, model: ModelType, query: str) -> None:
        """
        Validate that the query fits within the model's context window.

        This method estimates the number of tokens in the query and compares it
        to the maximum context size for the specified model. It logs a warning
        if the estimated token count exceeds the available context window.

        Args:
        ----
            model (ModelType): The model type to validate against.
            query (str): The input query string to be validated.

        Returns:
        -------
            None

        Side effects:
        ------------
            Logs a warning message if the estimated token count exceeds
            the model's context window.

        """
        estimated_tokens = len(query) // 4
        max_context = self.MODEL_CONTEXT_WINDOWS[model]

        if "online" in model:
            max_context -= 4096  # Reserve 4k tokens for search results

        if estimated_tokens > max_context:
            self.logger.warning(
                "Query may exceed model context window",
                extra={
                    "model": model,
                    "max_context": max_context,
                    "estimated_tokens": estimated_tokens,
                    "query_length": len(query),
                },
            )

    @abc.abstractmethod
    def search(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> Generator[SearchResponse, None, None] | AsyncGenerator[SearchResponse, None]:
        """
        Perform a search using the Perplexity API with streaming responses.

        This method sends a search query to the Perplexity API and returns a generator
        that yields search responses as they become available.

        Args:
        ----
            query (str): The search query string.
            mode (SearchMode, optional): The search mode to use. Defaults to "concise".
            search_focus (SearchFocus, optional): The focus area for the search. Defaults to "internet".
            timeout (float | None, optional): The maximum time to wait for a response, in seconds.
                                              If None, uses the default timeout. Defaults to None.
            model (ModelType, optional): The AI model to use for processing the query. Defaults to "sonar-pro".

        Returns:
        -------
            Generator[SearchResponse, None, None] | AsyncGenerator[SearchResponse, None]:
            A generator (synchronous or asynchronous) that yields search response objects.

        Raises:
        ------
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def search_sync(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> SearchResponse:
        """
        Perform a synchronous search using the Perplexity API and return a single response.

        This method sends a search query to the Perplexity API and returns a single, complete response.

        Args:
        ----
            query (str): The search query string to be processed.
            mode (SearchMode, optional): The search mode to use. Defaults to "concise".
            search_focus (SearchFocus, optional): The focus area for the search. Defaults to "internet".
            timeout (float | None, optional): The maximum time to wait for a response, in seconds.
                                              If None, uses the default timeout. Defaults to None.
            model (ModelType, optional): The AI model to use for processing the query. Defaults to "sonar-pro".

        Returns:
        -------
            SearchResponse: A dictionary containing the search results, typically including a 'text' key
                            with the response content.

        Raises:
        ------
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the client session and release any resources.

        This abstract method should be implemented by subclasses to properly
        close the client session, including any network connections or other
        resources that need to be released when the client is no longer needed.

        Raises
        ------
            NotImplementedError: This method must be implemented by subclasses.

        Returns
        -------
            None

        """
        raise NotImplementedError


class PerplexityClient(BasePerplexityClient):
    """Synchronous Perplexity AI client that uses API key authentication."""

    def __init__(
        self,
        api_key: str,
        *,
        api_version: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        logger: logging.Logger | None = None,
        default_model: str = "sonar-pro",
        base_url: str = "https://api.perplexity.ai",
    ) -> None:
        """
        Initialize the Perplexity AI client with the provided configuration.

        This method sets up the client with the given API key and configures the session
        with proper headers, retry strategy, and logging.

        Args:
        ----
            api_key (str): The API key for authentication with the Perplexity AI service.
            api_version (str | None, optional): The API version to use. If None, the default version will be used.
            max_retries (int, optional): The maximum number of retries for failed requests. Defaults to 3.
            timeout (float, optional): The default timeout for API requests in seconds. Defaults to 30.0.
            logger (logging.Logger | None, optional): A custom logger instance. If None, a default logger will be used.
            default_model (str, optional): The default model to use for API requests. Defaults to "sonar-pro".
            base_url (str, optional): The base URL for the Perplexity AI API. Defaults to "https://api.perplexity.ai".

        Returns:
        -------
            None

        Raises:
        ------
            AuthenticationError: If the provided API key is empty or invalid.

        """
        super().__init__(
            api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
            logger=logger,
            default_model=default_model,
            base_url=base_url,
        )

        # Configure session with proper headers and retry strategy
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"PerplexityClient/{self.CLIENT_VERSION}",
                "X-API-Version": self.api_version,
                "Accept": "application/json",
                "X-Request-ID": self._request_id,
            }
        )

        # Configure retry strategy with logging
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True,
            raise_on_status=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Log retries through event handler
        retry.get_backoff_time = lambda *args, **kwargs: (  # type: ignore
            self._log_retry(retry)
            or retry.__class__.get_backoff_time(retry, *args, **kwargs)
        )

    def search(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> Generator[SearchResponse, None, None]:
        """
        Perform a search using the Perplexity API with streaming responses.

        This method sends a search query to the Perplexity API and returns a generator
        that yields search responses as they become available.

        Args:
        ----
            query (str): The search query string.
            mode (SearchMode, optional): The search mode to use. Defaults to "concise".
            search_focus (SearchFocus, optional): The focus area for the search. Defaults to "internet".
            timeout (float | None, optional): The maximum time to wait for a response, in seconds.
                                              If None, uses the default timeout. Defaults to None.
            model (ModelType, optional): The AI model to use for processing the query. Defaults to "sonar-pro".

        Returns:
        -------
            Generator[SearchResponse, None, None]: A generator that yields search response objects.
                                                   Each response is a dictionary containing the 'text' key
                                                   with the response content.

        Raises:
        ------
            InvalidParameterError: If any of the input parameters are invalid.
            PerplexityTimeoutError: If the request times out.
            SearchError: If there's an error during the search process.

        """
        if not query:
            raise InvalidParameterError("query", "", ["non-empty string"])
        if mode not in self.VALID_MODES:
            raise InvalidParameterError(
                "mode", mode, [str(m) for m in self.VALID_MODES]
            )
        if search_focus not in self.VALID_SEARCH_FOCUS:
            raise InvalidParameterError(
                "search_focus",
                search_focus,
                [str(f) for f in self.VALID_SEARCH_FOCUS],
            )
        if model not in self.VALID_MODELS:
            raise InvalidParameterError(
                param="model", value=model, valid_values=list(self.VALID_MODELS)
            )

        try:
            self._log_request_start("search")

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": query}],
                "stream": True,
            }

            with self.session.post(
                f"{self.base_url}{self.CHAT_ENDPOINT}",
                json=payload,
                timeout=timeout or self.default_timeout,
                stream=True,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        try:
                            data = json.loads(line[6:])
                            if data.get("choices"):
                                yield {
                                    "text": data["choices"][0]["delta"].get(
                                        "content", ""
                                    )
                                }
                        except json.JSONDecodeError as e:
                            raise SearchError(query, mode, e)

            self._log_request_success("search")

        except requests.RequestException as e:
            self._log_request_failure("search", e)
            if isinstance(e, requests.Timeout):
                raise PerplexityTimeoutError(query, timeout or self.default_timeout)
            raise SearchError(query, mode, e)

    def search_sync(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> SearchResponse:
        """
        Perform a synchronous search using the Perplexity API and return a single response.

        This method sends a search query to the Perplexity API and returns a complete response
        without streaming.

        Args:
        ----
            query (str): The search query string to be processed.
            mode (SearchMode, optional): The search mode to use. Defaults to "concise".
            search_focus (SearchFocus, optional): The focus area for the search. Defaults to "internet".
            timeout (float | None, optional): The maximum time to wait for a response, in seconds.
                                              If None, uses the default timeout. Defaults to None.
            model (ModelType, optional): The AI model to use for processing the query. Defaults to "sonar-pro".

        Returns:
        -------
            SearchResponse: A dictionary containing the search results, typically including a 'text' key
                            with the response content.

        Raises:
        ------
            PerplexityTimeoutError: If the request times out.
            SearchError: If there's an error during the search process.

        """
        try:
            self._log_request_start("search_sync")

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": query}],
            }

            response = self.session.post(
                f"{self.base_url}{self.CHAT_ENDPOINT}",
                json=payload,
                timeout=timeout or self.default_timeout,
            )
            response.raise_for_status()

            data = response.json()
            result = {"text": data["choices"][0]["message"]["content"]}

            self._log_request_success("search_sync")
            return result

        except requests.RequestException as e:
            self._log_request_failure("search_sync", e)
            if isinstance(e, requests.Timeout):
                raise PerplexityTimeoutError(query, timeout or self.default_timeout)
            raise SearchError(query, mode, e)

    def close(self) -> None:
        """
        Close the client session and release associated resources.

        This method terminates the current session, closing any open connections
        and freeing up resources. It should be called when the client is no longer
        needed to ensure proper cleanup.

        Returns
        -------
        None

        """
        self.session.close()

    def __enter__(self) -> "PerplexityClient":
        """
        Enter the runtime context related to this object.

        This method allows the PerplexityClient to be used as a context manager.
        It is called when entering a 'with' statement.

        Returns
        -------
        PerplexityClient
            The PerplexityClient instance itself, allowing it to be used within a context.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit the runtime context related to this object.

        This method is called when exiting a 'with' statement. It ensures that
        the client's resources are properly released, regardless of whether an
        exception was raised within the 'with' block.

        Args:
        ----
            exc_type (type[BaseException] | None): The type of the exception that caused the context to be exited.
                                                   None if the context was exited without an exception.
            exc_val (BaseException | None): The instance of the exception that caused the context to be exited.
                                            None if the context was exited without an exception.
            exc_tb (TracebackType | None): A traceback object encapsulating the call stack at the point
                                           where the exception originally occurred.
                                           None if the context was exited without an exception.

        Returns:
        -------
            None

        """
        self.close()
