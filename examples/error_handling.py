"""
Showcases detailed error handling in PyPlexityAI.
"""

from pyplexityai import PerplexityClient
from pyplexityai.errors import (
    AuthenticationError,
    InvalidParameterError,
    PerplexityTimeoutError,
    SearchError,
)


def main():
    """
    Demonstrates error handling in PyPlexityAI by performing a search and handling various exceptions.

    This function creates a PerplexityClient, performs a synchronous search query,
    and handles potential errors that may occur during the process. It showcases
    how to properly catch and handle different types of exceptions that the
    PyPlexityAI library might raise.

    The function doesn't take any parameters and doesn't return any value.
    Instead, it prints the search result or error messages to the console.

    Exceptions handled:
    - AuthenticationError: Raised when there's an issue with the API key.
    - InvalidParameterError: Raised when an invalid parameter is provided.
    - SearchError: Raised when there's an error during the search process.
    - PerplexityTimeoutError: Raised when the request times out.

    Note:
    ----
    The client is properly closed in the 'finally' block to ensure
    resource cleanup, regardless of whether an exception occurred or not.

    """
    client = PerplexityClient("your-api-key")

    try:
        result = client.search_sync("What is quantum computing?")
        print("Search successful:", result["text"])
    except AuthenticationError as e:
        print(f"Authentication error: {e.message}")
    except InvalidParameterError as e:
        print(f"Invalid parameter: {e.message}")
    except SearchError as e:
        print(f"Search error: {e.message}")
        print(f"Possible causes: {e.causes}")
    except PerplexityTimeoutError as e:
        print(f"Request timed out: {e.message}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
