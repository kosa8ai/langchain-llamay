"""llamay chat models."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.language_models import LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama
from pydantic import Field, model_validator
from typing_extensions import Self

from langchain_llamay._base import default_base_url
from langchain_llamay._client import LlamayAsyncClient, LlamayClient


def _llamay_tool_choice(choice: dict | str | bool | None) -> dict | str | None:
    """LangChain's tool_choice in a form llamay reads.

    llamay takes OpenAI's strings and object and Anthropic's object. LangChain
    adds True for "call something" and a bare tool name for "call this one".
    """
    if choice is None or choice is False:
        return None
    if choice is True:
        return "required"
    if isinstance(choice, dict):
        return choice
    if choice in ("auto", "none", "any", "required"):
        return choice
    return {"type": "function", "function": {"name": choice}}


class ChatLlamay(ChatOllama):
    """Chat with a model served by `llamay serve`.

    llamay serves Ollama's API, so this is `ChatOllama` pointed at llamay, with
    one addition Ollama does not have: `tool_choice` is honoured. llamay
    constrains generation to a call when one is required, so even a small
    local model calls the tool it is told to.

    Setup:
        Install llamay from https://llamay.com/download/ and start a server:

        .. code-block:: bash

            llamay serve -m qwen2.5:7b
            pip install -U langchain-llamay

    Instantiate:
        .. code-block:: python

            from langchain_llamay import ChatLlamay

            llm = ChatLlamay(model="qwen2.5:7b")
            llm.invoke("Why is the sky blue?")

    Tool calling:
        .. code-block:: python

            llm.bind_tools([get_weather], tool_choice="get_weather").invoke("Hi")

    Key init args:
        model: str
            Any name `llamay models` lists, or a Hugging Face reference such as
            ``hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M``.
        base_url: str | None
            The server. Defaults to ``LLAMAY_HOST``, else
            ``http://localhost:11435``.
    """

    base_url: str | None = Field(default_factory=default_base_url)

    @model_validator(mode="after")
    def _llamay_clients(self) -> Self:
        # ChatOllama has built its clients by now, with every setting it was
        # given; giving them the subclass that can send tool_choice keeps all
        # of that rather than rebuilding it here.
        if self._client is not None:
            self._client.__class__ = LlamayClient
        if self._async_client is not None:
            self._async_client.__class__ = LlamayAsyncClient
        return self

    @property
    def _llm_type(self) -> str:
        return "chat-llamay"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "llamay"
        # A per-call model= is the model that answered; ChatOllama reports the
        # constructor's, which files the trace under the wrong model.
        params["ls_model_name"] = kwargs.get("model", self.model)
        return params

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "any"] | bool | None = None,  # noqa: PYI051
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools, and optionally which of them the model must call.

        Args:
            tools: Tool definitions, in any form ``convert_to_openai_tool`` reads.
            tool_choice: ``"auto"``, ``"none"``, ``"any"`` or ``True`` (call one
                of them), a tool's name (call that one), or an OpenAI or
                Anthropic tool_choice object.
            kwargs: Passed to ``self.bind``.
        """
        choice = _llamay_tool_choice(tool_choice)
        if choice is not None:
            kwargs["tool_choice"] = choice
        # Not super().bind_tools: its own tool_choice parameter would take the
        # choice and drop it, which is what this method exists to stop.
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _chat_params(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        # Kept out of ChatOllama's params, which go to the ollama client as
        # they are, and put back for LlamayClient.chat to send.
        choice = kwargs.pop("tool_choice", None)
        params = super()._chat_params(messages, stop, **kwargs)
        if choice is not None:
            params["tool_choice"] = choice
        return params
