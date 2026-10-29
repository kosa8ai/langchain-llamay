"""Standard tests against a live `llamay serve -m qwen2.5:0.5b -embed all-minilm:latest`."""

import os

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_llamay import ChatLlamay

MODEL = os.environ.get("LLAMAY_TEST_MODEL", "qwen2.5:0.5b")


class TestChatLlamayIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatLlamay]:
        return ChatLlamay

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL, "temperature": 0}

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return False
