# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TuxWhisper is private, offline push-to-talk dictation for Linux (X11 and Wayland, GNOME first).
Pure Python package in `tuxwhisper/`, a small Rust key-press client in `client-rs/`, shell
installer scripts. Published on PyPI as `tuxwhisper`; version lives in `tuxwhisper/__init__.py`.

## Commands

```bash
uvx ruff@0.16.10 check                 # lint (ruff.toml; CI pins this version)
.venv/bin/python -m pytest -q          # all tests (no mic, GPU or desktop needed)
.venv/bin/python -m pytest -q tests/test_engine.py -k cpu_fallback   # one file / one test
cd client-rs && cargo build --release  # Rust client; tests/test_key_client.py skips without it
cd client-rs && cargo fmt --check && cargo clippy --release -- -D warnings
shellcheck asr install.sh uninstall.sh
./asr <command>                        # run from the clone (same commands as `tuxwhisper`)
./asr serve                            # the daemon in the foreground
systemctl --user restart tuxwhisper    # the installed daemon runs this clone's ./asr
```

CI (`.github/workflows/ci.yml`) runs ruff, cargo fmt/clippy/build, pytest on Python 3.11/3.12/3.14,
shellcheck, then installs on several distros in containers. Release: bump `__version__`,
commit "Release X", push, wait for CI, tag `vX`, `gh release create`; the Publish workflow
uploads to PyPI.

## Architecture

Two processes talk over a Unix socket in `$XDG_RUNTIME_DIR`:

- **Daemon** (`asr serve`): `server.serve()` holds a lock, binds the socket and hands each
  connection to `Dictation` (`daemon.py`): the mic, the key press/hold logic, the model
  manager. A stopped take goes through `pipeline.TakePipeline` (transcribe, optional Ollama
  cleanup in `llm.py`, replacements, then `Paster`: clipboard + a virtual keyboard paste via
  `/dev/uinput`, then clipboard restore). Long takes are transcribed in parts by
  `streaming.py` (LocalAgreement) while still recording.
- **Key presses** (`asr toggle|rewrite|save`): the `asr` wrapper runs the Rust client if
  built, else `python -S -m tuxwhisper`, which imports only `client.py` and `store.py` and
  writes the word to the socket. **The press path is latency-critical**: `store.py` and
  `client.py` import heavier modules lazily and must not grow top-level imports; `settings.py`
  exists so `store.py` stays light. Measure with `python -X importtime` before adding imports there.
- **Engine** (`engine.py`): `load_whisper` picks faster-whisper (CUDA/CPU) or whisper.cpp
  (Vulkan). `ModelManager` steps down the model ladder on CUDA OOM, ends on `small (CPU)`,
  pre-checks `room_short()` with nvidia-smi so a model known not to fit is never read from
  disk, and restores the preferred model when memory frees. Idle unload after N minutes.
- **Menu/CLI** (`menu.py`, `asr.py`): settings menu via yad (preferred) or zenity, model
  picker, correct-last-transcript, Quit. All dialogs go through `desktop.zenity`/`desktop.notify`
  so tests can fake them; never call `subprocess.run(["zenity", ...])` directly (tests hang
  opening a real dialog).
- **Settings**: `Settings.from_config` merges `ASR_*` environment defaults with
  `~/.config/tuxwhisper/config.toml`. User files: `store.CONFIG_DIR` (config, vocab,
  replacements, modes), `store.DATA_DIR/recordings`, runtime files in `$XDG_RUNTIME_DIR`.

## Tests

`tests/conftest.py` has an autouse fixture that points every user file at a temp dir and stubs
notifications, sounds and history. Fakes: `FakeStream` (mic), `FakeModel` (transcribes
"take1", "take2"...), `fake_gpu(monkeypatch, free_gb, loads, cpu=True)` for the OOM ladder.
When adding a dialog or subprocess call in daemon/menu code, give it a seam in `desktop.py`
and patch that in tests.

## Project conventions

- Commit messages: one line, describe the change, no trailers of any kind. Commit, push and
  release only when asked.
- `notes.md`, `samples/`, `claude.txt` and the review `*.md` files are gitignored working
  files; never commit them. Don't put machine IPs or hardware specs in the repo.
- Docs in `docs/` are user-facing and shipped (`docs/pypi.md` is the PyPI readme); update
  `docs/configuration.md` / `docs/troubleshooting.md` when behaviour users see changes.
- Comments explain why (a measurement, a desktop quirk), not what; keep that style.
- Container desktop tests: one at a time, delete each after.
