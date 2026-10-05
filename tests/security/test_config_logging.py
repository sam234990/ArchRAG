"""Small, offline regression tests; no experiment dependencies are required."""

import argparse
import copy
import json
import unittest

from src.experiment_logging import experiment_config
from src.graphrag.config_logging import REDACTED, format_config, safe_config


class ConfigLoggingTests(unittest.TestCase):
    def test_secret_aliases_and_nested_containers(self):
        aliases = (
            "api_key", "embedding_api_key", "WANDB_API_KEY", "apiKey",
            "api-key", "access_token", "azureAdToken", "client_secret",
            "password", "credentials", "Authorization", "headers",
            "connection_string", "private_key", "aws_secret_access_key",
            "api_base", "embedding_api_base", "proxy", "storage_account_blob_url",
        )
        for name in aliases:
            with self.subTest(field=name):
                data = argparse.Namespace(nested=[({name: "FAKE-secret-value"},)])
                output = safe_config(data)
                self.assertEqual(output["nested"][0][0][name], REDACTED)
                self.assertNotIn("FAKE-secret-value", json.dumps(output))

    def test_normal_experiment_parameters_remain_useful(self):
        data = dict(model="synthetic-model", max_tokens=17, tokens_per_minute=400,
                    max_community_tokens=80, temperature=0.1, debug_flag=False)
        self.assertEqual(safe_config(data), data)

    def test_redaction_does_not_mutate_or_share_nested_config(self):
        data = {"llm": {"api_key": "FAKE-chat", "temperature": 0.2},
                "steps": [{"embedding_api_key": "FAKE-embed"}]}
        before = copy.deepcopy(data)
        output = safe_config(data)
        output["llm"]["temperature"] = 0.9
        self.assertEqual(data, before)

    def test_unknown_objects_and_non_string_keys_are_not_stringified(self):
        class SecretObject:
            def __repr__(self):
                raise AssertionError("must not call repr")

            __str__ = __repr__

        output = safe_config({"extra": SecretObject(), SecretObject(): "FAKE-secret"})
        self.assertEqual(output, {"extra": REDACTED})

    def test_recursive_config_is_bounded(self):
        data = {"api_key": "FAKE-cycle"}
        data["self"] = data
        output = format_config(data)
        self.assertNotIn("FAKE-cycle", output)
        self.assertIn(REDACTED, output)

    def test_sensitive_values_are_hidden_regardless_of_type(self):
        for value in (None, 123, ["FAKE-secret"], {"value": "FAKE-secret"}):
            self.assertEqual(safe_config({"api_key": value}), {"api_key": REDACTED})

    def test_wandb_allowlist_excludes_new_fields_paths_and_credentials(self):
        args = argparse.Namespace(
            dataset_name="synthetic", engine="dummy", max_tokens=12,
            api_key="FAKE-chat", embedding_api_key="FAKE-embed",
            output_dir="/private/data", api_base="https://FAKE-url.invalid",
            future_option="FAKE-new-secret", nested={"api_key": "FAKE-nested"},
        )
        before = copy.deepcopy(vars(args))
        self.assertEqual(experiment_config(args),
                         {"dataset_name": "synthetic", "engine": "dummy", "max_tokens": 12})
        self.assertEqual(vars(args), before)

    def test_wandb_rejects_unexpected_nested_values_in_allowed_fields(self):
        self.assertEqual(experiment_config({"engine": {"api_key": "FAKE-secret"}}), {})


if __name__ == "__main__":
    unittest.main()
