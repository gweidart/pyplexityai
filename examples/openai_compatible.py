"""
Demonstrates usage of the OpenAI-compatible interface in PyPlexityAI.
"""

import asyncio

from pyplexityai import OpenAICompatibleClient
from pyplexityai.client_types import ChatMessage


def sync_openai_compatible():
    """
    A function demonstrating a synchronous request to the OpenAI-compatible interface in PyPlexityAI.
    It sends a chat message asking for an explanation of quantum mechanics and streams the tokens back.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example:
    >>> sync_openai_compatible()
    Sync OpenAI-Compatible Streaming:
    Explanation of quantum mechanics...
    ---

    """
    client = OpenAICompatibleClient("your-api-key")

    messages = [
        ChatMessage(role="user", content="Explain quantum mechanics", name=None)
    ]

    # Stream tokens in a synchronous OpenAI-compatible request
    print("Sync OpenAI-Compatible Streaming:")
    for response in client.create_chat_completion(
        messages=messages,
        model="sonar-pro",
        stream=True,
    ):
        if "delta" in response["choices"][0]:
            if content := response["choices"][0]["delta"].get("content"):
                print(content, end="", flush=True)
    client.close()  # type: ignore
    print("\n---\n")


async def async_openai_compatible():
    """
    Demonstrates an asynchronous request to the OpenAI-compatible interface in PyPlexityAI.

    This function sends a chat message asking for an explanation of quantum entanglement
    and streams the tokens back asynchronously.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Example:
    >>> asyncio.run(async_openai_compatible())
    Async OpenAI-Compatible Streaming:
    Explanation of quantum entanglement...

    """
    async with OpenAICompatibleClient("your-api-key") as client:
        messages = [
            ChatMessage(role="user", content="Explain quantum entanglement", name=None)
        ]

        print("Async OpenAI-Compatible Streaming:")
        async for response in client.acreate_chat_completion(
            messages=messages,
            model="sonar-pro",
            stream=True,
        ):
            if "delta" in response["choices"][0]:
                if content := response["choices"][0]["delta"].get("content"):
                    print(content, end="", flush=True)


def main():
    sync_openai_compatible()
    asyncio.run(async_openai_compatible())


if __name__ == "__main__":
    main()
