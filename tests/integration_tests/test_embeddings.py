from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_llamay import LlamayEmbeddings


class TestLlamayEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> type[LlamayEmbeddings]:
        return LlamayEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "all-minilm:latest"}
