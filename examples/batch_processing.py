"""
Demonstrates batch processing of queries using the async client.
"""

import asyncio

from pyplexityai import AsyncPerplexityClient
from pyplexityai.errors import PerplexityTimeoutError


async def batch_search(queries: list[str], api_key: str) -> dict[str, str]:
    results = {}
    async with AsyncPerplexityClient(api_key) as client:
        for query in queries:
            try:
                # Adjust timeout as desired for bulk queries
                data = await client.async_search_sync(query, timeout=45.0)
                results[query] = data["text"]
            except PerplexityTimeoutError:
                results[query] = "Error: timeout"
    return results  # type: ignore


async def main():
    queries = ["What is Python?", "What is JavaScript?", "What is Rust?"]

    results = await batch_search(queries, "your-api-key")

    for query, result in results.items():
        print(f"\nQuery: {query}\nResult: {result}")


if __name__ == "__main__":
    asyncio.run(main())
