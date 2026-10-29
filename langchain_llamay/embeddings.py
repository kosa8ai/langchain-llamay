"""llamay embeddings."""

from langchain_ollama import OllamaEmbeddings
from pydantic import Field

from langchain_llamay._base import default_base_url


class LlamayEmbeddings(OllamaEmbeddings):
    """Embeddings from the encoder a llamay server was started with.

    llamay serves embeddings from an encoder loaded beside the chat model:

    .. code-block:: bash

        llamay serve -m qwen2.5:7b -embed all-minilm:latest

    .. code-block:: python

        from langchain_llamay import LlamayEmbeddings

        embeddings = LlamayEmbeddings(model="all-minilm:latest")
        embeddings.embed_query("hello")
    """

    base_url: str | None = Field(default_factory=default_base_url)
