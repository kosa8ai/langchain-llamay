from langchain_tests.unit_tests import EmbeddingsUnitTests

from langchain_llamay import LlamayEmbeddings


class TestLlamayEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> type[LlamayEmbeddings]:
        return LlamayEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "all-minilm:latest"}


def test_embeddings_default_to_llamays_port(monkeypatch) -> None:
    monkeypatch.delenv("LLAMAY_HOST", raising=False)
    assert LlamayEmbeddings(model="all-minilm:latest").base_url == "http://localhost:11435"
