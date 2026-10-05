"""Exercise logging boundaries with fake data and no GPU/LLM/network access.

ArchRAG entry points are imported with heavy dependencies stubbed. GraphRAG
logging functions are loaded from their actual source bodies in isolation from
the pipeline/decorators; this deliberately does not test a full GraphRAG import.
"""

import argparse
import ast
import asyncio
import copy
import importlib.util
from functools import cache
import io
import logging
import os
from pathlib import Path
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from src.graphrag import config_logging

ROOT = Path(__file__).resolve().parents[2]
SECRETS = ("FAKE-chat-secret", "FAKE-embedding-secret", "FAKE-header-secret")


def module_stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def load_module(path, name, stubs):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs), patch.dict(os.environ):
        spec.loader.exec_module(module)
    return module


def load_functions(path, names, **bindings):
    """Keep real function bodies and sanitizer imports; isolate heavy imports."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if
                 isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
                 or isinstance(node, ast.ImportFrom) and node.module == "graphrag.config_logging")
    namespace = {"verb": lambda **kw: lambda fn: fn, **bindings}
    with patch.dict(sys.modules, {"graphrag.config_logging": config_logging}):
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     str(ROOT / path), "exec"), namespace)
    return types.SimpleNamespace(**namespace)


class StopBeforeData(Exception):
    pass


class LoggingSinkTests(unittest.TestCase):
    def setUp(self):
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            guard = patch(target, side_effect=AssertionError("network forbidden in security tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def assert_no_secrets(self, value):
        for secret in SECRETS:
            self.assertNotIn(secret, str(value))

    def fake_config(self):
        return {"llm": {"api_key": SECRETS[0], "max_tokens": 23},
                "steps": [{"embedding_api_key": SECRETS[1]}],
                "headers": {"Authorization": SECRETS[2]}}

    def load_utils(self):
        stubs = {name: module_stub(name) for name in
                 ("src.lm_emb", "faiss", "numpy", "scipy", "scipy.spatial")}
        stubs.update({
            "tiktoken": module_stub("tiktoken", Encoding=type("Encoding", (), {})),
            "tqdm": module_stub("tqdm", tqdm=Mock()),
            "networkx": module_stub("networkx", Graph=type("Graph", (), {})),
            "pandas": module_stub("pandas", DataFrame=type("DataFrame", (), {})),
            "json_repair": module_stub("json_repair", repair_json=Mock()),
            "scipy.spatial.distance": module_stub("scipy.spatial.distance", cosine=Mock()),
        })
        return load_module("src/utils.py", "security_test_utils", stubs)

    def test_print_args_with_real_parsers_and_nested_config(self):
        utils = self.load_utils()
        for parser in (utils.create_arg_parser(), utils.create_inference_arg_parser()):
            args = parser.parse_args([])
            args.api_key, args.embedding_api_key = SECRETS[:2]
            args.extra = self.fake_config()
            before = copy.deepcopy(vars(args))
            output = io.StringIO()
            with redirect_stdout(output):
                utils.print_args(args)
                utils.print_args(self.fake_config(), "Synthetic parameters:")
            self.assert_no_secrets(output.getvalue())
            self.assertIn("max_tokens", output.getvalue())
            self.assertEqual(vars(args), before)

    def test_actual_wandb_call_receives_only_public_config(self):
        wandb = module_stub("wandb", init=Mock(), Settings=lambda **kw: kw)
        stop = Mock(side_effect=StopBeforeData)
        stubs = {
            "wandb": wandb,
            "pandas": module_stub("pandas", DataFrame=type("DataFrame", (), {})),
            "src.inference": module_stub("src.inference", load_index=stop),
            "src.utils": module_stub("src.utils", create_inference_arg_parser=Mock()),
            "src.evaluate.evaluate": module_stub("src.evaluate.evaluate"),
        }
        module = load_module("src/evaluate/test_qa.py", "security_test_qa", stubs)
        args = argparse.Namespace(project="synthetic", dataset_name="synthetic", seed=7,
                                  api_key=SECRETS[0], embedding_api_key=SECRETS[1],
                                  extra=self.fake_config())
        with self.assertRaises(StopBeforeData):
            module.test_qa({"generate_strategy": "mr"}, args)
        wandb.init.assert_called_once()
        payload = wandb.init.call_args.kwargs
        self.assert_no_secrets(payload)
        self.assertEqual(payload["config"], {"dataset_name": "synthetic", "seed": 7})
        self.assertTrue(payload["settings"]["x_disable_meta"])
        self.assertTrue(payload["settings"]["disable_git"])
        self.assertTrue(payload["settings"]["disable_code"])
        self.assertFalse(payload["settings"]["save_code"])
        self.assertEqual(payload["settings"]["console"], "off")
        self.assertEqual(args.api_key, SECRETS[0])

    def test_graphrag_cli_and_llm_settings_log_nested_config(self):
        data = self.fake_config()
        cli = load_functions("src/graphrag/index/cli.py", {"redact"})
        self.assert_no_secrets(cli.redact(data))
        logger = logging.getLogger("security_test_settings")
        funcs = load_functions("src/graphrag/index/create_pipeline_config.py", {"_log_llm_settings"}, log=logger)
        llm = types.SimpleNamespace(model_dump=lambda: data)
        settings = types.SimpleNamespace(entity_extraction=types.SimpleNamespace(llm=llm),
                                         embeddings=types.SimpleNamespace(llm=llm))
        with self.assertLogs(logger, "INFO") as logs:
            funcs._log_llm_settings(settings)
        self.assert_no_secrets(logs.output)
        self.assertIn("23", str(logs.output))

    def test_strategy_debug_logs_before_any_data_processing(self):
        locations = (
            ("entities/extraction/entity_extract.py", "entity_extract"),
            ("entities/summarize/description_summarize.py", "summarize_descriptions"),
            ("graph/report/create_community_reports.py", "create_community_reports"),
            ("covariates/extract_covariates/extract_covariates.py", "extract_covariates"),
        )
        for path, name in locations:
            with self.subTest(sink=name):
                logger = logging.getLogger("security_test_" + name)
                funcs = load_functions("src/graphrag/index/verbs/" + path, {name}, log=logger,
                                       AsyncType=types.SimpleNamespace(AsyncIO=None),
                                       DEFAULT_ENTITY_TYPES=[], cast=lambda typ, val: val,
                                       pd=types.SimpleNamespace(DataFrame=object))
                data = Mock()
                data.get_input.side_effect = StopBeforeData
                with self.assertLogs(logger, "DEBUG") as logs, self.assertRaises(StopBeforeData):
                    asyncio.run(getattr(funcs, name)(input=data, cache=None, callbacks=None,
                                column="text", id_column="id", to="out", covariate_type="claim",
                                strategy=self.fake_config()))
                self.assert_no_secrets(logs.output)

    def test_query_factories_hide_credentials_but_pass_them_to_client(self):
        llm_types = types.SimpleNamespace(AzureOpenAIChat="azure_chat", AzureOpenAI="azure",
                                         AzureOpenAIEmbedding="azure_embedding")
        client = Mock()
        funcs = load_functions("src/graphrag/query/factories.py", {"get_llm", "get_text_embedder"},
                               LLMType=llm_types, ChatOpenAI=client, OpenAIEmbedding=client,
                               OpenaiApiType=types.SimpleNamespace(OpenAI="openai", AzureOpenAI="azure"))
        llm = types.SimpleNamespace(type="openai", api_key=SECRETS[0], api_base=SECRETS[1],
                                    cognitive_services_endpoint=None, organization=None,
                                    model="dummy", deployment_name=None, api_version=None,
                                    max_retries=0, model_dump=self.fake_config)
        config = types.SimpleNamespace(llm=llm, embeddings=types.SimpleNamespace(llm=llm))
        output = io.StringIO()
        with redirect_stdout(output):
            funcs.get_llm(config)
            funcs.get_text_embedder(config)
        self.assert_no_secrets(output.getvalue())
        self.assertEqual(client.call_count, 2)
        self.assertEqual(client.call_args.kwargs["api_key"], SECRETS[0])
        self.assertEqual(client.call_args.kwargs["api_base"], SECRETS[1])

    def test_vector_store_reporter_hides_nested_config(self):
        reporter = Mock()
        funcs = load_functions("src/graphrag/query/api.py", {"local_search"},
                               reporter=reporter, validate_call=lambda **kw: lambda fn: fn,
                               VectorStoreType=types.SimpleNamespace(LanceDB="lancedb"),
                               read_indexer_entities=Mock(side_effect=StopBeforeData))
        data = self.fake_config()
        config = types.SimpleNamespace(embeddings=types.SimpleNamespace(vector_store=data))
        with self.assertRaises(StopBeforeData):
            asyncio.run(funcs.local_search(config, None, None, None, None, None, None, 1, "QA", "dummy"))
        reporter.info.assert_called_once()
        self.assert_no_secrets(reporter.info.call_args)
        self.assertEqual(config.embeddings.vector_store, data)

    def test_client_creation_does_not_log_credential_bearing_endpoints(self):
        class Config:
            api_key = SECRETS[0]
            api_base = "https://user:" + SECRETS[1] + "@example.invalid/path?key=" + SECRETS[2]
            deployment_name = SECRETS[2]
            cognitive_services_endpoint = organization = api_version = None
            request_timeout = 1

        logger = logging.getLogger("security_test_client")
        client = Mock()
        funcs = load_functions("src/graphrag/llm/openai/create_openai_client.py", {"create_openai_client"},
                               cache=cache, log=logger, AsyncOpenAI=client, AsyncAzureOpenAI=client)
        with self.assertLogs(logger, "INFO") as logs:
            funcs.create_openai_client(Config(), False)
            funcs.create_openai_client(Config(), True)
        self.assert_no_secrets(logs.output)
        self.assertEqual(client.call_args.kwargs["api_key"], SECRETS[0])
        self.assertEqual(client.call_args.kwargs["azure_endpoint"], Config.api_base)


if __name__ == "__main__":
    unittest.main()
