"""
Basic asynchronous example of querying PerplexityAI using PyPlexityAI.
"""

import asyncio

from pyplexityai import AsyncPerplexityClient


async def main():
    """
    Asynchronously queries PerplexityAI for information about Python.

    import asyncio
        This function demonstrates the basic usage of AsyncPerplexityClient
        to perform an asynchronous search query.

    from pyplexityai import AsyncPerplexityClient

    Returns:
        None: This function doesn't return any value, but prints the search result.

    """
    async with AsyncPerplexityClient("your-api-key") as client:
        result = await client.async_search_sync("What is Python?")
        print("Async Result:", result["text"])


if __name__ == "__main__":
    asyncio.run(main())
