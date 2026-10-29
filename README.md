# langchain-llamay

LangChain integration for [llamay](https://llamay.com), a local LLM runtime for
macOS, Windows and Linux.

```bash
pip install -U langchain-llamay
```

## Start a server

Download llamay from [llamay.com/download](https://llamay.com/download/), then:

```bash
llamay serve -m qwen2.5:7b
```

llamay downloads a model it does not have yet, including GGUF repositories on
Hugging Face (`hf.co/<owner>/<repo>:<quant>`), and listens on
`http://localhost:11435`. Set `LLAMAY_HOST` to reach a server elsewhere.

## Chat

```python
from langchain_llamay import ChatLlamay

llm = ChatLlamay(model="qwen2.5:7b")
print(llm.invoke("Why is the sky blue?").content)
```

Tool calling, structured output and streaming work as they do in
`langchain-ollama`, which this package builds on: llamay serves Ollama's API.

One thing Ollama lacks does work here: `tool_choice`. llamay constrains the
answer to a call when one is required, so even a small model calls the tool it
is told to.

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


llm.bind_tools([get_weather], tool_choice="get_weather").invoke("Hello!").tool_calls
# [{'name': 'get_weather', 'args': {'city': ...}, ...}]
```

`tool_choice` takes `"auto"`, `"none"`, `"any"` or `True` (call one of the
tools), a tool's name, or an OpenAI or Anthropic tool_choice object.

```python
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str


llm.with_structured_output(City).invoke("Name a city in Japan.")
```

## Embeddings

Start the server with an encoder beside the chat model:

```bash
llamay serve -m qwen2.5:7b -embed all-minilm:latest
```

```python
from langchain_llamay import LlamayEmbeddings

embeddings = LlamayEmbeddings(model="all-minilm:latest")
vector = embeddings.embed_query("hello")
```

## Tests

```bash
uv run --group test pytest tests/unit_tests
# with `llamay serve -m qwen2.5:0.5b -embed all-minilm:latest` running:
uv run --group test pytest tests/integration_tests
```
