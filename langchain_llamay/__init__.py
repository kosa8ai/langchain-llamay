"""LangChain integration for llamay, a local LLM runtime."""

from importlib import metadata

from langchain_llamay.chat_models import ChatLlamay
from langchain_llamay.embeddings import LlamayEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = ["ChatLlamay", "LlamayEmbeddings", "__version__"]
