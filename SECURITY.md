# Safe use of ArchRAG

ArchRAG is research software for experiments with trusted code, configuration,
datasets and artifacts. It has not been hardened as a service for untrusted
users, network peers or uploaded index files. These precautions also apply to
clones and forks used for reproduction or development.

## Credentials and experiment tracking

- Keep API keys out of committed scripts, configuration examples, notebook
  outputs, run names, dataset/model labels and issue reports. Use private local
  configuration or read credentials from the environment inside Python.
  ArchRAG's argument parsers do not automatically read environment variables:
  assign the values to `args.api_key` and `args.embedding_api_key` in a local
  Python entry point. Expanding an environment variable into `--api_key` still
  exposes its value in process arguments and potentially shell history.
- `print_args` and the covered vendored GraphRAG configuration logs use
  `safe_config` from `src/graphrag/config_logging.py`. It copies nested mappings,
  namespaces and sequences, hides credential fields and endpoints/proxies, and
  does not modify the credentials used by the experiment. Unknown objects are
  hidden without calling their `str` or `repr` methods.
- `src/evaluate/test_qa.py` passes only the reviewed scalar fields in
  `src/experiment_logging.py` to W&B. New CLI fields are excluded by default.
  This entry point also disables W&B automatic system metadata (which can
  include command-line arguments), Git/source-code and console capture. The
  `x_disable_meta` setting is specific to the pinned W&B version; recheck it
  when upgrading W&B. Explicit experiment parameters and metrics can still be
  sent to W&B. Set `WANDB_MODE=disabled` before launching an experiment if no
  tracking is wanted. Offline W&B runs can retain data for later synchronization.
- These filters are for structured configuration, not arbitrary text. They do
  not identify secrets embedded in otherwise public labels, prompts, dataset
  contents, exception messages, or logs produced by other code/dependencies.
  Review outputs before sharing. Runtime `model_dump()` / `model_dump_json()`
  intentionally retain credentials for execution and must not be logged or
  uploaded directly. Do not pass whole configuration objects to new log sinks.
- This change does not inspect or delete existing logs or W&B runs, or rotate
  any keys. If a credential is known to have been disclosed, its owner should
  revoke/rotate it and review the affected outputs separately. The presence of
  a risky code path alone is not evidence that a disclosure occurred.

## Optional pickle RPC examples: trusted peers only

`HCHNSW/faiss/contrib/rpc.py` and `client_server.py` are optional research
examples, not a secure serving API. They serialize messages with pickle,
provide no trustworthy authentication/encryption boundary, and expose callable
operations on the server/index. The `RestrictedUnpickler` name and its NumPy
module allowlist must **not** be treated as a security boundary.

**Do not expose these examples to untrusted networks or clients.** The current
server binds a wildcard address; launching it on a workstation or shared host
can make it reachable beyond that process. Do not publish its port, enable
port-forwarding, or put it on a shared LAN assuming it is safe. Use only trusted
clients and servers in an isolated environment whose network access is enforced
outside the example. A client must also trust the server's serialized replies.

This warning does not fix the protocol, change its bind address, restrict its
RPC methods, or establish safe deserialization. Those changes require separate
design and testing.

## Load only trusted indexes and caches

- Load FAISS/C-HNSW indexes only from a source you trust, preferably generated
  locally with a known build. The custom native index parser is not a sandbox
  for files supplied by strangers. This applies to `faiss.read_index` and the
  ArchRAG index loaders even when a file has an expected extension.
- Pickle checkpoints (including the optional FAISS big-batch search checkpoint)
  and NumPy object caches loaded with `allow_pickle=True` can reconstruct Python
  objects. Do not load untrusted copies. For example,
  `src/evaluate/hchnsw_evaluate.py` uses an object-capable ground-truth cache.
- Keep experiment directories writable only by trusted users. Verify artifact
  provenance through a trusted channel; a checksum supplied alongside an
  untrusted file does not make its contents safe.

These are usage restrictions, not repairs to native parsing or pickle loading.

## Scope and remaining work

The first batch updates GitPython from 3.1.44 to the upstream
[3.2.0 security release](https://github.com/gitpython-developers/GitPython/releases/tag/3.2.0),
adds structured configuration redaction and a W&B metadata allowlist, and
documents the trust assumptions above. Numerical/GPU dependencies are unchanged.

Remaining work includes review and repair of the custom FAISS C++ parsers,
replacing/hardening the optional RPC protocol and dispatch, safer cache formats,
and triage of other dependency/static-analysis findings. The previously reported
15 Security Cloud findings came from a partial static scan, not successful
runtime exploitation tests. This batch does not establish that all findings are
fixed or that any secrets have leaked. Re-scan and review each finding after a
maintainer has accepted the changes.

## Focused offline regression checks

From the repository root, without installing the experiment stack:

```bash
python -m unittest discover -s tests/security -v
```

The basic tests use fake credentials, small in-memory configurations, mocked
clients and a mocked W&B call. They stop before data loading or model execution.
GraphRAG logging functions are tested from their source bodies in isolation from
heavy pipeline imports/decorators. Optional model/dependency checks are skipped
if their dependencies are absent. These checks do not validate the full pipeline.

To include real Pydantic models and the GitPython/W&B local metadata smoke test,
use a disposable environment (Python 3.10 is the project's target):

```bash
python3.10 -m venv /tmp/archrag-security-venv
/tmp/archrag-security-venv/bin/python -m pip install --index-url https://pypi.org/simple \
  -c requirements.txt GitPython==3.2.0 gitdb==4.0.12 wandb==0.19.10
/tmp/archrag-security-venv/bin/python -m unittest discover -s tests/security -v
/tmp/archrag-security-venv/bin/python -m pip check
```

Package installation needs network access. The tests themselves use no real
datasets, paid LLMs, GPU build, W&B login or W&B upload. The dependency smoke test
creates and removes a tiny local Git repository and checks W&B's Git metadata
adapter plus capture settings. Do not use a repository-wide test collection as
a substitute: the repository also contains experiment scripts and native FAISS
tests with different requirements. Full experiments, native parser attack tests
and online W&B behavior must be validated separately if needed.
