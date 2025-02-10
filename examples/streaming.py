"""
Examples of streaming responses synchronously and asynchronously.
"""

import asyncio

from pyplexityai import AsyncPerplexityClient, PerplexityClient


def sync_stream_example():
    with PerplexityClient("your-api-key") as client:
        print("Sync Stream Example:")
        for chunk in client.search(
            "How does quantum computing differ from classical computing?"
        ):
            if "text" in chunk:
                print(chunk["text"], end="", flush=True)
    print("\n---\n")


async def async_stream_example():
    async with AsyncPerplexityClient("your-api-key") as client:
        print("Async Stream Example:")
        async for chunk in client.async_search(
            "Explain quantum mechanics in layman terms"
        ):
            if "text" in chunk:
                print(chunk["text"], end="", flush=True)


def main():
    sync_stream_example()
    asyncio.run(async_stream_example())


if __name__ == "__main__":
    main()
