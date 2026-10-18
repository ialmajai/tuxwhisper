"""Shared fixtures and fakes: every user file in a temp dir, no mic, GPU or desktop."""
import threading
import types

import numpy as np
import pytest

from tuxwhisper import daemon, desktop, engine, llm, store

real_add_history = store.add_history  # the fixture below stubs it out for the dictation tests


real_notify = desktop.notify


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every user file at a temp dir and silence desktop side effects."""
    monkeypatch.setattr(store, "CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(store, "REPLACEMENTS", str(tmp_path / "replacements.txt"))
    monkeypatch.setattr(store, "VOCAB", str(tmp_path / "vocab.txt"))
    monkeypatch.setattr(store, "RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.setattr(store, "MODES_FILE", str(tmp_path / "modes.txt"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))  # history, retype job: never the real ones
    monkeypatch.delenv("ASR_MODEL", raising=False)
    monkeypatch.setattr(desktop, "sound", lambda name: None)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: None)
    monkeypatch.setattr(store, "add_history", lambda text, **kw: None)
    monkeypatch.setattr(desktop, "notify", lambda msg, **kw: notified.append(msg))
    notified.clear()


notified = []


class FakeStream:
    fail_start = False

    def __init__(self, callback, **_):
        self.callback = callback

    def start(self):
        if FakeStream.fail_start:
            raise OSError("Device unavailable")
        self.callback(np.full((store.RATE, 1), 0.1, dtype=np.float32))  # one second of "audio"

    def stop(self):
        pass

    def close(self):
        pass


class FakeModel:
    """Transcribes takes in order as 'take1', 'take2'...; the first one can be made slow."""

    def __init__(self):
        self.count = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.release_first.set()

    is_loaded = True

    def loaded(self):
        return self.is_loaded

    def unload(self, to_cpu=True):
        self.is_loaded = False

    def load(self):
        self.is_loaded = True

    def detect_language(self, audio, **_):
        assert self.is_loaded, "transcribing with the model unloaded"
        return ("en", 1.0)

    def transcribe(self, audio, **_):
        self.count += 1
        n = self.count
        if n == 1:
            self.first_started.set()
            self.release_first.wait(5)
        return [types.SimpleNamespace(text=f"take{n}")], None


def make_dictation(monkeypatch):
    d = daemon.Dictation(engine.ModelManager(FakeModel()), types.SimpleNamespace(InputStream=FakeStream), kbd=None, ec=None)
    d.pasted = []
    d.paster.paste = lambda text, snapshot=None: d.pasted.append(text.strip()) or True
    d.refresh_hint = lambda: None
    monkeypatch.setattr(llm, "warm_rewrite_model", lambda: None)
    monkeypatch.setattr(llm, "llm_cleanup", lambda text, lang, on_gpu=None: f"CLEANED:{text}")
    FakeStream.fail_start = False
    return d


class InlineThread:
    """threading.Thread that runs its target on start(), so a test sees the result at once."""
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class OOMModel(FakeModel):
    """A model parked in RAM that no longer fits back on the GPU."""
    is_loaded = False

    def load(self):
        raise RuntimeError("CUDA failed with error out of memory")


def fake_gpu(monkeypatch, free_gb, loads):
    """Pretend the GPU has free_gb free: bigger models fail to load as out of memory. Models
    that load are appended to loads."""
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    monkeypatch.setattr(engine, "free_gpu_gb", lambda: free_gb[0])
    monkeypatch.setattr(engine, "model_downloaded", lambda name: True)
    def load(name):
        if engine.model_size(name) > free_gb[0]:
            raise RuntimeError("CUDA failed with error out of memory")
        loads.append(name)
        return FakeModel()
    monkeypatch.setattr(engine, "load_whisper", load)


class FakeKeyboard:
    def __init__(self):
        self.keys = []

    def write(self, kind, code, val):
        if val:
            self.keys.append(code)

    def syn(self):
        pass


def burst(seconds, total=4.0, level=0.01, rate=16000):
    """A take of total seconds: faint noise with seconds of louder sound in the middle."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.0002, int(total * rate)).astype(np.float32)
    start = int((total - seconds) / 2 * rate)
    a[start:start + int(seconds * rate)] += rng.normal(0, level, int(seconds * rate)).astype(np.float32)
    return a


class WhisperedModel(FakeModel):
    """Like Whisper on a whispered take: the voice filter finds nothing; without it, `heard`."""
    def __init__(self, heard):
        super().__init__()
        self.heard = heard

    def transcribe(self, audio, vad_filter=True, **_):
        return ([] if vad_filter else [types.SimpleNamespace(text=self.heard)]), None


class RunOOMModel(FakeModel):
    """Loaded, but running it fails: another program took the GPU memory it needs."""
    def detect_language(self, audio, **_):
        raise RuntimeError("CUDA failed with error out of memory")

    def transcribe(self, audio, **_):
        raise RuntimeError("CUDA failed with error out of memory")


class LanguageModel(FakeModel):
    """Reports the language it detects, like faster-whisper's info.language."""
    def __init__(self, lang):
        super().__init__()
        self.lang, self.calls = lang, []

    def detect_language(self, audio, **_):
        raise AssertionError("detected separately")

    def transcribe(self, audio, language=None, prompt=None, **_):
        self.calls.append((language, prompt is not None))
        return [types.SimpleNamespace(text="hi")], types.SimpleNamespace(language=language or self.lang)
