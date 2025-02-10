"""Async version of the Perplexity AI client."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from types import TracebackType

try:
    import aiohttp
except ImportError as e:
    raise ImportError(
        "Async support requires additional dependencies. "
        "Please install with 'pip install perplexity-client[async]'"
    ) from e

from .client import BasePerplexityClient, SearchResponse
from .client_types import ModelType, SearchFocus, SearchMode
from .errors import InvalidParameterError, PerplexityTimeoutError, SearchError

logger = logging.getLogger("perplexity_client.async")


class AsyncPerplexityClient(BasePerplexityClient):
    """Async version of the Perplexity AI client."""

    def __init__(
        self,
        api_key: str,
        *,
        api_version: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        logger: logging.Logger | None = None,
        default_model: str = "sonar-pro",
    ) -> None:
        """Initialize the async client.

        Args:
        ----
            api_key: Your Perplexity AI API key
            api_version: Optional API version override (default: latest stable)
            max_retries: Maximum number of retries for failed requests
            timeout: Default timeout for requests in seconds
            logger: Optional logger instance for custom logging
            default_model: Default model to use for response generation

        Raises:
        ------
            AuthenticationError: If the API key is invalid or missing
            ImportError: If async dependencies are not installed

        """
        super().__init__(
            api_key,
            api_version=api_version,
            max_retries=max_retries,
            timeout=timeout,
            logger=logger,
            default_model=default_model,
        )
        self._session: aiohttp.ClientSession | None = None

    async def _create_session(self) -> aiohttp.ClientSession:
        """
        Create or return an existing aiohttp session with proper configuration.

        This method initializes a new aiohttp.ClientSession if one doesn't exist
        or if the existing session is closed. The session is configured with
        a timeout and headers including authentication, user agent, API version,
        and request ID.

        Returns
        -------
        aiohttp.ClientSession: An active aiohttp session configured for
            making requests to the Perplexity AI API.

        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": f"PerplexityClient/{self.CLIENT_VERSION}",
                    "X-API-Version": self.api_version,
                    "Accept": "application/json",
                    "X-Request-ID": self._request_id,
                },
            )

        return self._session

    def search(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> AsyncGenerator[SearchResponse, None]:
        """
        Asynchronous search function that yields results as they arrive.

        This method is not implemented and raises a NotImplementedError.
        Use async_search instead for asynchronous search functionality.

        Parameters
        ----------
        query : str
            The search query string.
        mode : SearchMode, optional
            The search mode to use (default is "concise").
        search_focus : SearchFocus, optional
            The focus of the search (default is "internet").
        timeout : float | None, optional
            The timeout for the search request in seconds (default is None).
        model : ModelType, optional
            The model to use for the search (default is "sonar-pro").

        Returns
        -------
        AsyncGenerator[SearchResponse, None]
            An asynchronous generator that yields search responses.

        Raises
        ------
        NotImplementedError
            This method is not implemented and will always raise this error.

        """
        raise NotImplementedError("Use async_search instead")

    async def async_search(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> AsyncGenerator[SearchResponse, None]:
        """
        Perform an asynchronous search and yield results as they arrive.

        This method sends a search query to the Perplexity AI API and streams the results
        back asynchronously. It validates input parameters before making the API request.

        Parameters
        ----------
        query : str
            The search query string to be sent to the API.
        mode : SearchMode, optional
            The search mode to be used. Default is "concise".
        search_focus : SearchFocus, optional
            The focus area for the search. Default is "internet".
        timeout : float | None, optional
            The maximum time (in seconds) to wait for the API response. If None,
            the default timeout will be used.
        model : ModelType, optional
            The AI model to be used for generating the response. Default is "sonar-pro".

        Returns
        -------
        AsyncGenerator[SearchResponse, None]
            An asynchronous generator that yields SearchResponse objects containing
            the search results as they are received from the API.

        Raises
        ------
        InvalidParameterError
            If any of the input parameters are invalid or not within the allowed values.
        PerplexityTimeoutError
            If the API request times out.
        SearchError
            If there's an error during the search process or in parsing the API response.

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
                "model", str(model), [str(m) for m in self.VALID_MODELS]
            )

        session = await self._create_session()

        try:
            self._log_request_start("search")

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": query}],
                "stream": True,
            }

            async with session.post(
                f"{self.API_BASE}{self.CHAT_ENDPOINT}",
                json=payload,
                timeout=timeout or self.default_timeout,
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.strip()
                    if not line or not line.startswith(b"data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                        if data.get("choices"):
                            yield {
                                "text": data["choices"][0]["delta"].get("content", "")
                            }
                    except json.JSONDecodeError as e:
                        raise SearchError(query, mode, e)

            self._log_request_success("search")

        except aiohttp.ClientError as e:
            self._log_request_failure("search", e)
            if isinstance(e, asyncio.TimeoutError):
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
        Perform a synchronous search and return the final result.

        This method is not implemented in the async client. Use async_search_sync instead.

        Parameters
        ----------
        query : str
            The search query string to be sent to the API.
        mode : SearchMode, optional
            The search mode to be used (default is "concise").
        search_focus : SearchFocus, optional
            The focus area for the search (default is "internet").
        timeout : float | None, optional
            The maximum time (in seconds) to wait for the API response. If None,
            the default timeout will be used.
        model : ModelType, optional
            The AI model to be used for generating the response (default is "sonar-pro").

        Returns
        -------
        SearchResponse
            The final search result containing the generated response.

        Raises
        ------
        NotImplementedError
            This method is not implemented in the async client.

        """
        raise NotImplementedError("Use async_search_sync instead")

    async def async_search_sync(
        self,
        query: str,
        mode: SearchMode = "concise",
        search_focus: SearchFocus = "internet",
        timeout: float | None = None,
        model: ModelType = "sonar-pro",
    ) -> SearchResponse:
        """
        Perform an asynchronous search and return the final result synchronously.

        This method sends a search query to the Perplexity AI API and returns the complete
        response as a single result, rather than streaming it.

        Parameters
        ----------
        query : str
            The search query string to be sent to the API.
        mode : SearchMode, optional
            The search mode to be used. Default is "concise".
        search_focus : SearchFocus, optional
            The focus area for the search. Default is "internet".
        timeout : float | None, optional
            The maximum time (in seconds) to wait for the API response. If None,
            the default timeout will be used.
        model : ModelType, optional
            The AI model to be used for generating the response. Default is "sonar-pro".

        Returns
        -------
        SearchResponse
            A dictionary containing the search result with a 'text' key holding the
            generated response content.

        Raises
        ------
        PerplexityTimeoutError
            If the API request times out.
        SearchError
            If there's an error during the search process or in parsing the API response.

        """
        session = await self._create_session()

        try:
            self._log_request_start("search_sync")

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": query}],
            }

            async with session.post(
                f"{self.API_BASE}{self.CHAT_ENDPOINT}",
                json=payload,
                timeout=timeout or self.default_timeout,
            ) as response:
                response.raise_for_status()
                data = await response.json()

            result = {"text": data["choices"][0]["message"]["content"]}
            self._log_request_success("search_sync")
            return result

        except aiohttp.ClientError as e:
            self._log_request_failure("search_sync", e)
            if isinstance(e, asyncio.TimeoutError):
                raise PerplexityTimeoutError(query, timeout or self.default_timeout)
            raise SearchError(query, mode, e)

    def close(self) -> None:
        """
        Close the session.

        This method is not implemented in the async client. Use async_close instead.

        Raises
        ------
        NotImplementedError
            This method is not implemented and will always raise this error.

        """
        raise NotImplementedError("Use async_close instead")

    async def async_close(self) -> None:
        """
        Asynchronously close the client session.

        This method closes the aiohttp ClientSession if it exists. It's important to call
        this method when you're done using the client to properly release resources and
        close any open connections.

        Returns:
        -------
        None
            This method doesn't return anything.

        Note:
        ----
        After calling this method, the client's session will be set to None, and you'll
        need to create a new session before making any further requests.

        """
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncPerplexityClient":
        """
        Asynchronous context manager entry method.

        This method is called when entering an async context manager block using the
        'async with' statement. It ensures that a session is created before the context
        is entered.

        Returns
        -------
        AsyncPerplexityClient
            Returns the instance of the AsyncPerplexityClient, allowing it to be used
            within the context manager block.

        """
        await self._create_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Asynchronous context manager exit method.

        This method is called when exiting an async context manager block. It ensures
        that the client session is properly closed, releasing any resources and
        connections.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            The type of the exception that caused the context to be exited, or None if no exception was raised.
        exc_val : BaseException | None
            The instance of the exception that caused the context to be exited, or None if no exception was raised.
        exc_tb : TracebackType | None
            A traceback object encapsulating the call stack at the point where the exception was raised, or None if no exception was raised.

        Returns
        -------
        None
            This method doesn't return anything.

        """
        await self.async_close()
