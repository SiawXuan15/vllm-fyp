# Mac development environment

This environment supports source editing, linting and preparing PD experiments.
It does not provide a validated vLLM runtime or the Linux NVIDIA NIXL baseline.

From the repository root, activate it in each new terminal:

```bash
source research/prefill_pd/scripts/activate-dev.sh
```

The helper adds ~/.local/bin (where uv is installed) to this shell's PATH,
activates the existing .venv, and clears vLLM build overrides from this shell.
It does not edit shell startup files or recreate the environment.

Verified locally: Python 3.12.7, pre-commit 4.7.0, and installed dependency
consistency with `uv pip check --python .venv/bin/python`. The existing virtual
environment was created from an Anaconda Python but excludes system packages;
there is no need to replace it solely for that reason.

Useful commands:

```bash
uv pip check --python .venv/bin/python
.venv/bin/pre-commit validate-config
.venv/bin/pre-commit run shellcheck --files research/prefill_pd/scripts/serve-b0.sh research/prefill_pd/scripts/bench-b0.sh research/prefill_pd/scripts/activate-dev.sh
bash -n research/prefill_pd/scripts/serve-b0.sh research/prefill_pd/scripts/bench-b0.sh research/prefill_pd/scripts/activate-dev.sh
```

Both pre-commit and commit-msg Git hooks are installed. Other hook environments
may download dependencies on first use. Some repository checks require the Linux
runtime; a working lint environment does not establish that all vLLM tests can run.

Do not run the CUDA precompiled editable install on this Mac. The checked-out
`docs/getting_started/installation/cpu.apple.inc.md` describes a separate native
CPU source build, but that is unnecessary for the current PD preparation task.
Use installation instructions matching the pinned checkout on the GPU server.
