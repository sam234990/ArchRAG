"""Real configuration classes, with pipeline package initialization bypassed."""

from enum import Enum
import importlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

from test_logging_sinks import ROOT, load_module, module_stub
from src.graphrag import config_logging


class AsyncType(str, Enum):
    AsyncIO = "asyncio"
    Threaded = "threaded"


def package(name, path):
    result = types.ModuleType(name)
    result.__path__ = [str(ROOT / path)]
    return result


class ConfigModelTests(unittest.TestCase):
    def test_openai_config_representations_preserve_runtime_credentials(self):
        protocol = load_module("src/graphrag/llm/types/llm_config.py", "security_llm_protocol", {})
        module = load_module("src/graphrag/llm/openai/openai_configuration.py", "security_openai_config", {
            "graphrag.config_logging": config_logging,
            "graphrag.llm.types": module_stub("graphrag.llm.types", LLMConfig=protocol.LLMConfig),
        })
        raw = {"api_key": "FAKE-model-secret", "model": "dummy",
               "nested": [{"access_token": "FAKE-model-secret"}]}
        config = module.OpenAIConfiguration(raw)
        for output in (str(config), repr(config)):
            self.assertNotIn("FAKE-model-secret", output)
            self.assertIn("dummy", output)
        self.assertEqual(config.api_key, "FAKE-model-secret")
        self.assertIs(config.raw_config, raw)

    @unittest.skipUnless(importlib.util.find_spec("pydantic"), "optional: install the pinned lightweight dependencies")
    def test_real_pydantic_models_redact_repr_but_not_serialization(self):
        stubs = {
            "graphrag": package("graphrag", "src/graphrag"),
            "graphrag.config": package("graphrag.config", "src/graphrag/config"),
            "graphrag.index": package("graphrag.index", "src/graphrag/index"),
            "graphrag.config_logging": config_logging,
            "datashaper": module_stub("datashaper", AsyncType=AsyncType),
        }
        with patch.dict(sys.modules, stubs):
            graph_type = importlib.import_module("graphrag.config.models.graph_rag_config").GraphRagConfig
            pipeline_type = importlib.import_module("graphrag.index.config.pipeline").PipelineConfig
            configs = (
                graph_type(llm={"api_key": "FAKE-pydantic-secret"},
                           embeddings={"llm": {"api_key": "FAKE-pydantic-secret"},
                                       "vector_store": {"apiKey": "FAKE-pydantic-secret"}}),
                pipeline_type(workflows=[{"name": "synthetic", "config":
                              {"steps": [{"llm": {"api_key": "FAKE-pydantic-secret"}}]}}]),
            )
            for config in configs:
                with self.subTest(config_type=type(config).__name__):
                    self.assertNotIn("FAKE-pydantic-secret", str(config))
                    self.assertNotIn("FAKE-pydantic-secret", repr(config))
                    self.assertIn("FAKE-pydantic-secret", config.model_dump_json())


if __name__ == "__main__":
    unittest.main()
