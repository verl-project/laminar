# Contributing to Laminar

Keep each pull request focused on one change, describe the resulting behavior,
and include relevant validation. Use a title such as `[train] fix: describe the change`.
Report security issues privately as described in [SECURITY.md](SECURITY.md).

## Development

- **Environment:** Follow [README.md#getting-started](README.md#getting-started) (add `--extra dev` for test and lint tools).
- **Tests:** Run CPU unit tests for your change (e.g., with the `fsdp`, `vllm`, and `ucx` extras):
  ```bash
  uv run --locked --extra dev --extra fsdp --extra vllm --extra ucx python -m pytest -q tests/<test_file>
  ```
- **Formatting and lint:** Check modified files with `ruff check`, `ruff format --check`, and `git diff --check`. Keep Apache-2.0 headers on all `.py` and `.sh` files.
- **Dependencies:** Edit pins in `pyproject.toml`, then synchronize backend settings and lock:
  ```bash
  uv run --no-project scripts/sync_dependency_config.py
  uv lock
  ```
- **Release hygiene:** Do not include private endpoints, credentials, personal paths, or internal logs.
