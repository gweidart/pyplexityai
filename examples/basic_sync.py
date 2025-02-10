"""
Basic synchronous example of querying PerplexityAI using PyPlexityAI.
"""

from pyplexityai import PerplexityClient
from pyplexityai.errors import AuthenticationError


def main():
    """
    Demonstrates the usage of PerplexityClient for synchronous searches.

    This function showcases two methods of using the PerplexityClient:
    1. Direct instantiation with try-except-finally block
    2. Using a context manager (recommended approach)

    It performs two searches:
    1. "Explain quantum computing in simple terms"
    2. "What is quantum entanglement?"

    The function handles potential authentication errors and ensures
    proper closure of the client connection.

    Parameters
    ----------
    None

    Returns
    -------
    None. Results are printed to the console.

    """
    # Replace "your-api-key" with a valid key.
    client = PerplexityClient("your-api-key")

    try:
        # A simple synchronous search
        result = client.search_sync("Explain quantum computing in simple terms")
        print("Sync Result:", result["text"])
    except AuthenticationError as e:
        print(f"Authentication failed: {e}")
    finally:
        client.close()

    # Using the client as a context manager (recommended)
    with PerplexityClient("your-api-key") as client:
        result = client.search_sync("What is quantum entanglement?")
        print("Context Manager Result:", result["text"])


if __name__ == "__main__":
    main()
