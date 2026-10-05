"""Optional smoke checks for the pinned GitPython/W&B pair; local Git only."""

import importlib.util
from importlib.metadata import version
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("git") and importlib.util.find_spec("wandb"),
                     "optional: install the pinned lightweight dependencies")
class DependencySmokeTests(unittest.TestCase):
    def test_pinned_wandb_accepts_capture_restrictions(self):
        import wandb
        from src.experiment_logging import experiment_tracking_settings

        settings = experiment_tracking_settings(wandb)
        self.assertTrue(settings.x_disable_meta)
        self.assertTrue(settings.disable_git)
        self.assertTrue(settings.disable_code)
        self.assertFalse(settings.save_code)
        self.assertEqual(settings.console, "off")

    def test_gitpython_with_wandb_local_git_metadata(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network forbidden")), \
             patch("socket.getaddrinfo", side_effect=AssertionError("network forbidden")), \
             patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}), \
             tempfile.TemporaryDirectory() as directory:
            from git import Actor, Repo
            from wandb.sdk.lib.gitlib import GitRepo

            self.assertEqual(version("GitPython"), "3.2.0")
            self.assertEqual(version("gitdb"), "4.0.12")
            self.assertEqual(version("wandb"), "0.19.10")
            repo = Repo.init(directory)
            self.addCleanup(repo.close)
            source = Path(directory) / "synthetic.md"
            source.write_text("Synthetic fixture only.\n")
            repo.index.add([str(source)])
            actor = Actor("Security test", "security-test@example.invalid")
            commit = repo.index.commit("Synthetic local fixture", author=actor, committer=actor)
            metadata = GitRepo(root=directory)
            self.addCleanup(lambda: metadata.repo.close() if metadata.repo else None)
            self.assertEqual(metadata.last_commit, commit.hexsha)
            self.assertFalse(metadata.dirty)
            self.assertEqual(Path(metadata.root).resolve(), Path(directory).resolve())
            source.write_text("Changed synthetic fixture.\n")
            self.assertTrue(metadata.dirty)


if __name__ == "__main__":
    unittest.main()
