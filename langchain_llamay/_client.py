"""The ollama client, able to send llamay's tool_choice.

Ollama's /api/chat has no tool_choice, so the ollama client has no argument for
it. llamay's does: it reads the field in OpenAI's and Anthropic's shapes and
constrains generation to a call, which is how a small local model can be made
to call a tool at all. These subclasses add the field to the request body.

The choice travels in a context variable rather than on the client, because one
client serves every call a chat model makes, concurrent ones included. The body
is built inside chat() before it returns -- for a stream too, whose generator
only reads the response -- so the variable is still set when _request runs.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from ollama import AsyncClient, Client

_tool_choice: ContextVar[Any] = ContextVar("llamay_tool_choice", default=None)


def _with_choice(kwargs: dict[str, Any], args: tuple[Any, ...]) -> dict[str, Any]:
    choice = _tool_choice.get()
    if choice is not None and len(args) >= 2 and args[1] == "/api/chat" and "json" in kwargs:
        kwargs = {**kwargs, "json": {**kwargs["json"], "tool_choice": choice}}
    return kwargs


class LlamayClient(Client):
    def chat(self, *args: Any, tool_choice: Any = None, **kwargs: Any) -> Any:
        token = _tool_choice.set(tool_choice)
        try:
            return super().chat(*args, **kwargs)
        finally:
            _tool_choice.reset(token)

    def _request(self, cls: Any, *args: Any, **kwargs: Any) -> Any:
        return super()._request(cls, *args, **_with_choice(kwargs, args))


class LlamayAsyncClient(AsyncClient):
    async def chat(self, *args: Any, tool_choice: Any = None, **kwargs: Any) -> Any:
        token = _tool_choice.set(tool_choice)
        try:
            return await super().chat(*args, **kwargs)
        finally:
            _tool_choice.reset(token)

    async def _request(self, cls: Any, *args: Any, **kwargs: Any) -> Any:
        return await super()._request(cls, *args, **_with_choice(kwargs, args))
