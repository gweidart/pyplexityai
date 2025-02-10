"""Test suite for PyPlexityAI client."""

import asyncio
import os
import sys
from collections.abc import Callable
from typing import Any, cast

import requests
from dotenv import load_dotenv

from pyplexityai import (
    AsyncPerplexityClient,
    AuthenticationError,
    ChatMessage,
    InvalidParameterError,
    ModelType,
    OpenAICompatibleClient,
    PerplexityClient,
)

# Load environment variables
load_dotenv()
api_key = os.getenv("PERPLEXITY_API_KEY")

if not api_key:
    raise ValueError("PERPLEXITY_API_KEY environment variable not set")

# Map of test names to functions
TEST_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def test_basic_search():
    """
    Test basic search functionality with different models.

    This function performs a series of search queries using different Perplexity AI models.
    It tests the basic search functionality by sending predefined queries to each model
    and printing the responses. The function also handles potential authentication errors.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AuthenticationError: If there's an issue with the API key authentication.
    requests.exceptions.HTTPError: If there's an HTTP error during the API request.

    Note:
    This function requires a valid Perplexity API key to be set in the environment.

    """
    print("\n=== Testing Basic Search ===")

    with PerplexityClient(cast(str, api_key)) as client:
        # Test different models
        models: list[ModelType] = ["sonar-pro", "sonar-reasoning-pro", "sonar"]
        queries = [
            "Write a Python function to implement quicksort",
            "Explain the PAXOS consensus algorithm",
            "What are the key features of Rust's ownership system?",
        ]

        for model in models:
            print(f"\nTesting {model} model:")
            for query in queries:
                print(f"\nQuery: {query}")
                print("Response:")
                try:
                    for chunk in client.search(query, model=model):
                        if "text" in chunk:
                            print(chunk["text"], end="", flush=True)
                    print("\n" + "=" * 50)
                except (AuthenticationError, requests.exceptions.HTTPError) as e:
                    if (
                        isinstance(e, requests.exceptions.HTTPError)
                        and e.response.status_code == 401
                    ):
                        print(
                            "⚠️  Skipping test due to authentication error. Please check your API key."
                        )
                        return
                    if isinstance(e, AuthenticationError):
                        print(
                            "⚠️  Skipping test due to authentication error. Please check your API key."
                        )
                        return
                    raise


TEST_FUNCTIONS["basic"] = test_basic_search


def test_search_modes():
    """
    Test different search modes and focus areas using the PerplexityClient.

    This function demonstrates the usage of various search modes and focus areas
    available in the PerplexityClient. It performs two tests:
    1. A search using the 'copilot' mode for code generation.
    2. A search using the 'concise' mode with a 'scholar' focus.

    The function prints the results of each search and handles potential
    authentication errors.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AuthenticationError
        If there's an issue with the API key authentication.
    requests.exceptions.HTTPError
        If there's an HTTP error during the API request, other than a 401 status.

    """
    print("\n=== Testing Search Modes and Focus Areas ===")

    with PerplexityClient(cast(str, api_key)) as client:
        try:
            # Test copilot mode with code generation
            print("\nTesting copilot mode (code generation):")
            query = "Write a FastAPI endpoint that accepts a JSON payload and stores it in PostgreSQL"
            for chunk in client.search(query, mode="copilot", model="sonar-pro"):
                if "text" in chunk:
                    print(chunk["text"], end="", flush=True)

            # Test concise mode with scholar focus
            print("\n\nTesting concise mode with scholar focus:")
            query = "Latest research on quantum computing error correction"
            for chunk in client.search(
                query, mode="concise", search_focus="scholar", model="sonar-pro"
            ):
                if "text" in chunk:
                    print(chunk["text"], end="", flush=True)
        except (AuthenticationError, requests.exceptions.HTTPError) as e:
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response.status_code == 401
            ):
                print(
                    "⚠️  Skipping test due to authentication error. Please check your API key."
                )
                return
            if isinstance(e, AuthenticationError):
                print(
                    "⚠️  Skipping test due to authentication error. Please check your API key."
                )
                return
            raise


TEST_FUNCTIONS["modes"] = test_search_modes


async def test_async_client():
    """
    Test the asynchronous functionality of the Perplexity AI client.

    This function demonstrates the usage of the AsyncPerplexityClient by performing
    multiple asynchronous searches with predefined queries. It showcases the client's
    ability to handle concurrent requests and stream responses.

    The function uses a set of queries related to various technical topics and
    processes them using the 'sonar-pro' model. It prints both the queries and
    their corresponding responses.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AuthenticationError
        If there's an issue with the API key authentication.
    requests.exceptions.HTTPError
        If there's an HTTP error during the API request, other than a 401 status.

    Note
    ----
    This function requires a valid Perplexity API key to be set in the environment.
    It uses the global 'api_key' variable.

    """
    print("\n=== Testing Async Client ===")

    async with AsyncPerplexityClient(cast(str, api_key)) as client:
        queries = [
            "What are the best practices for Kubernetes pod security?",
            "Explain how BERT handles attention mechanisms",
            "Compare different microservices communication patterns",
        ]

        try:
            for query in queries:
                print(f"\nQuery: {query}")
                print("Response:")
                async for chunk in client.async_search(query, model="sonar-pro"):
                    if "text" in chunk:
                        print(chunk["text"], end="", flush=True)
                print("\n" + "=" * 50)
        except (AuthenticationError, requests.exceptions.HTTPError) as e:
            if (
                isinstance(e, requests.exceptions.HTTPError)
                and e.response.status_code == 401
            ):
                print(
                    "⚠️  Skipping test due to authentication error. Please check your API key."
                )
                return
            if isinstance(e, AuthenticationError):
                print(
                    "⚠️  Skipping test due to authentication error. Please check your API key."
                )
                return
            raise


TEST_FUNCTIONS["async"] = test_async_client


def test_openai_compatibility():
    """
    Test OpenAI compatibility layer with streaming and chat functionality.

    This function demonstrates the usage of the OpenAICompatibleClient by performing
    a multi-turn chat completion with predefined messages. It showcases the client's
    ability to handle streaming responses and complex conversations.

    The function tests two scenarios:
    1. A chat completion with system and user messages.
    2. A follow-up question in a multi-turn conversation.

    It also handles potential authentication errors and HTTP errors.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AuthenticationError:
        If there's an issue with the API key authentication.
    requests.exceptions.HTTPError:
        If there's an HTTP error during the API request, other than a 401 status.

    Note:
    -----
    This function requires a valid Perplexity API key to be set in the environment.
    It uses the global 'api_key' variable.

    """
    print("\n=== Testing OpenAI Compatibility ===")
    client = OpenAICompatibleClient(cast(str, api_key))

    try:
        # Test chat completion with system and user messages
        messages = [
            ChatMessage(
                role="system",
                content="You are an expert software architect with deep knowledge of distributed systems.",
                name=None,
            ),
            ChatMessage(
                role="user",
                content="Design a scalable event-driven architecture for a real-time analytics platform. Include technologies and patterns you would use.",
                name=None,
            ),
        ]

        print("\nStreaming chat completion:")
        for response in client.create_chat_completion(
            messages=messages,
            model="sonar-pro",
            stream=True,
        ):
            if "delta" in response["choices"][0]:
                if content := response["choices"][0]["delta"].get("content"):
                    print(content, end="", flush=True)

        # Test multi-turn conversation
        messages.append(
            ChatMessage(
                role="assistant",
                content="[Previous response about event-driven architecture]",
                name=None,
            )
        )
        messages.append(
            ChatMessage(
                role="user",
                content="How would you handle data consistency and fault tolerance in this design?",
                name=None,
            )
        )

        print("\n\nFollow-up question:")
        for response in client.create_chat_completion(
            messages=messages,
            model="sonar-pro",
            stream=True,
        ):
            if "delta" in response["choices"][0]:
                if content := response["choices"][0]["delta"].get("content"):
                    print(content, end="", flush=True)
    except (AuthenticationError, requests.exceptions.HTTPError) as e:
        if (
            isinstance(e, requests.exceptions.HTTPError)
            and e.response.status_code == 401
        ):
            print(
                "⚠️  Skipping test due to authentication error. Please check your API key."
            )
            return
        if isinstance(e, AuthenticationError):
            print(
                "⚠️  Skipping test due to authentication error. Please check your API key."
            )
            return
        raise


TEST_FUNCTIONS["openai"] = test_openai_compatibility


def test_error_handling():
    """
    Test error handling and edge cases in the PerplexityClient.

    This function performs a series of tests to ensure proper error handling
    in various scenarios, including:
    1. Authentication errors with an invalid API key.
    2. Invalid search mode errors.
    3. Invalid model errors.

    The function attempts to trigger these errors and verifies that they
    are caught and handled correctly.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    Any unexpected exceptions that are not caught during the error handling tests.

    Note:
    -----
    This function relies on the global `api_key` variable for valid client tests.

    """
    print("\n=== Testing Error Handling ===")

    # Test authentication error
    try:
        with PerplexityClient("invalid-key") as client:
            client.search_sync("test")
    except (AuthenticationError, requests.exceptions.HTTPError) as e:
        if (
            isinstance(e, requests.exceptions.HTTPError)
            and e.response.status_code == 401
        ):
            print("✓ Successfully caught authentication error")
        elif isinstance(e, AuthenticationError):
            print("✓ Successfully caught authentication error")

    # Test other errors with valid client
    with PerplexityClient(cast(str, api_key)) as client:
        # Test invalid mode
        try:
            client.search_sync("", mode="invalid_mode")  # type: ignore
        except InvalidParameterError:
            print("✓ Successfully caught invalid parameter error")

        # Test invalid model
        try:
            client.search_sync("test", model="nonexistent-model")  # type: ignore
        except InvalidParameterError:
            print("✓ Successfully caught invalid model error")


TEST_FUNCTIONS["errors"] = test_error_handling


async def run_selected_tests(test_names: list[str] | None = None) -> None:
    """
    Run selected tests or all tests if none specified.

    This function executes a series of tests for the PyPlexityAI Test Suite. It can run all available tests
    or a subset of tests based on the provided test names. The function handles both synchronous and
    asynchronous test functions.

    Parameters
    ----------
    test_names : list[str] | None, optional
        A list of test names to run. If None (default), all available tests will be executed.
        Valid test names are keys in the TEST_FUNCTIONS dictionary.

    Returns
    -------
    None
        This function doesn't return any value but prints the test results to the console.

    Raises
    ------
    Exception
        Any exception that occurs during the execution of tests is caught, logged, and re-raised.

    Notes
    -----
    - The function prints start and end messages for the test suite.
    - If no test names are provided, it runs all predefined tests.
    - For specified test names, it checks if each test exists before running it.
    - The 'async' test is handled differently due to its asynchronous nature.
    - Test results and any errors are printed to the console.

    """
    try:
        print("\n🚀 Starting PyPlexityAI Test Suite")
        print("=" * 50)

        if not test_names:
            # Run all tests
            test_basic_search()
            test_search_modes()
            await test_async_client()
            test_openai_compatibility()
            test_error_handling()
        else:
            # Run only selected tests
            for name in test_names:
                if name not in TEST_FUNCTIONS:
                    print(f"⚠️  Unknown test: {name}")
                    continue

                print(f"\nRunning test: {name}")
                if name == "async":
                    await cast(Callable[[], Any], TEST_FUNCTIONS[name])()
                else:
                    TEST_FUNCTIONS[name]()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e!s}")
        raise


if __name__ == "__main__":
    # Get test names from command line arguments
    test_names = sys.argv[1:] if len(sys.argv) > 1 else None

    if test_names:
        print(f"Running selected tests: {', '.join(test_names)}")
    else:
        print("Running all tests")

    asyncio.run(run_selected_tests(test_names))
