from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_llamay import ChatLlamay
from langchain_llamay._base import DEFAULT_BASE_URL


class TestChatLlamayUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatLlamay]:
        return ChatLlamay

    @property
    def chat_model_params(self) -> dict:
        return {"model": "qwen2.5:0.5b"}


def test_defaults_to_llamays_port(monkeypatch) -> None:
    monkeypatch.delenv("LLAMAY_HOST", raising=False)
    assert ChatLlamay(model="qwen2.5:0.5b").base_url == DEFAULT_BASE_URL == "http://localhost:11435"


def test_reads_llamay_host(monkeypatch) -> None:
    monkeypatch.setenv("LLAMAY_HOST", "http://gpu-box:11435")
    assert ChatLlamay(model="qwen2.5:0.5b").base_url == "http://gpu-box:11435"


def test_an_explicit_base_url_wins(monkeypatch) -> None:
    monkeypatch.setenv("LLAMAY_HOST", "http://gpu-box:11435")
    assert ChatLlamay(model="m", base_url="http://other:1").base_url == "http://other:1"


def test_traces_as_llamay() -> None:
    llm = ChatLlamay(model="qwen2.5:0.5b")
    assert llm._llm_type == "chat-llamay"
    assert llm._get_ls_params()["ls_provider"] == "llamay"


def test_tool_choice_reaches_the_request(monkeypatch) -> None:
    # ChatOllama drops tool_choice; ChatLlamay sends it, in the shape llamay
    # reads, on the same /api/chat request as the tools.
    sent: list[dict] = []

    def fake_request(self, cls, *args, **kwargs):
        sent.append(kwargs.get("json", {}))
        return cls(model="m", message={"role": "assistant", "content": ""}, done=True)

    import ollama

    monkeypatch.setattr(ollama.Client, "_request", fake_request)

    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "sunny"

    llm = ChatLlamay(model="qwen2.5:0.5b")
    for choice, want in [
        ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
        ("any", "any"),
        (True, "required"),
        ("auto", "auto"),
    ]:
        sent.clear()
        llm.bind_tools([get_weather], tool_choice=choice).invoke("hi", stream=False)
        assert sent and sent[-1].get("tool_choice") == want, (choice, sent)
        assert sent[-1]["tools"][0]["function"]["name"] == "get_weather"

    sent.clear()
    llm.bind_tools([get_weather]).invoke("hi", stream=False)
    assert "tool_choice" not in sent[-1]
