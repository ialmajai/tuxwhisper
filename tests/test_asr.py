"""Tests for TuxWhisper's pure helpers and the dictation core, without a mic, GPU or desktop."""
import os
import stat
import subprocess
import sys
import threading
import time
import tomllib
import types

import numpy as np
import pytest

from tuxwhisper import asr


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every user file at a temp dir and silence desktop side effects."""
    monkeypatch.setattr(asr, "CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(asr, "REPLACEMENTS", str(tmp_path / "replacements.txt"))
    monkeypatch.setattr(asr, "VOCAB", str(tmp_path / "vocab.txt"))
    monkeypatch.setattr(asr, "RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.setattr(asr, "sound", lambda name: None)
    monkeypatch.setattr(asr, "close_notification", lambda nid: None)
    monkeypatch.setattr(asr, "add_history", lambda text: None)
    monkeypatch.setattr(asr, "notify", lambda msg, **kw: notified.append(msg))
    notified.clear()


notified = []


# --- keys -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, binding", [
    ("F6", "F6"), ("f12", "F12"), ("Ctrl+Alt+D", "<Control><Alt>d"), ("super+space", "<Super>space"),
    ("Alt+F9", "<Alt>F9"), ("Shift+Insert", "<Shift>Insert"),
    ("d", None), ("space", None), ("Hyper+X", None), ("F25", None), ("Ctrl+", None), ("", None),
])
def test_parse_key(text, binding):
    assert asr.parse_key(text) == binding


@pytest.mark.parametrize("binding, label", [
    ("<Alt>F9", "Alt+F9"), ("<Control><Alt>d", "Ctrl+Alt+D"), ("<Primary>h", "Ctrl+H"),
    ("F5", "F5"), ("<Super>space", "Super+Space"), ("", ""),
])
def test_key_label(binding, label):
    assert asr.key_label(binding) == label


def test_key_suggestions_round_trip():
    for label in asr.KEY_SUGGESTIONS:
        assert asr.key_label(asr.parse_key(label)) == label



def test_fit_sizes_list_dialogs():
    rows = zip(["F5", "F3"], ["Dictation", "Cleanup"])  # callers pass iterators
    width, height = asr.fit(("Key", "Action"), rows)
    assert width.startswith("--width=") and height.startswith("--height=")
    long_row = asr.fit(("Value",), [("x" * 120,)])[0]
    assert int(long_row.split("=")[1]) > int(width.split("=")[1])  # long cells widen the dialog


def test_yad_args_translate_zenity_options():
    assert asr.yad_args(("--list", "--ok-label=Copy", "--column=A")) == [
        "--list", "--column=A", "--button=Cancel:1", "--button=Copy:0"]
    assert asr.yad_args(("--error", "--text=x")) == ["--image=dialog-error", "--text=x", "--button=OK:0"]
    assert asr.yad_args(("--question", "--text=x"))[-2:] == ["--button=No:1", "--button=Yes:0"]

# --- replacements ---------------------------------------------------------------------------

def test_replacements(tmp_path):
    (tmp_path / "replacements.txt").write_text(
        "# comment => ignored\nnew line => \\n\ncomma => ,\ntux whisper => TuxWhisper\n")
    assert asr.apply_replacements("Hello. New line. World.") == "Hello.\nWorld."
    assert asr.apply_replacements("I like tux whisper") == "I like TuxWhisper"
    assert asr.apply_replacements("commander") == "commander"  # whole words only


def test_replacements_missing_file():
    assert asr.apply_replacements("unchanged") == "unchanged"


# --- settings -------------------------------------------------------------------------------

def test_toml_value_round_trips():
    cfg = {"sounds": True, "sound_volume": 57, "language": "", "mode": 'Say "hi"',
           "mode_models": {"Email": "qwen3:8b", 'My "q" mode': "x"}}
    asr.save_settings(cfg)
    with open(asr.CONFIG, "rb") as f:
        assert tomllib.load(f) == cfg


def test_settings_reject_bad_values():
    with open(asr.CONFIG, "w") as f:
        f.write('sound_volume = "loud"\npaste_key = "alt+v"\nlanguage = 7\nmode_models = "x"\n')
    s = asr.settings()
    assert s["sound_volume"] == 40
    assert s["paste_key"] == "shift+insert"
    assert s["language"] == ""  # a wrong-typed language would reach Whisper
    assert s["mode_models"] == {}


def test_settings_clamp_volume():
    with open(asr.CONFIG, "w") as f:
        f.write("sound_volume = 250\n")
    assert asr.settings()["sound_volume"] == 100


def test_bad_env_volume_does_not_crash_import(tmp_path):
    env = dict(os.environ, ASR_SOUND_VOLUME="abc", XDG_CONFIG_HOME=str(tmp_path))
    r = subprocess.run([sys.executable, "-c", "import tuxwhisper.asr as a; print(a.settings()['sound_volume'])"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "40"


# --- dictation core -------------------------------------------------------------------------

class FakeStream:
    fail_start = False

    def __init__(self, callback, **_):
        self.callback = callback

    def start(self):
        if FakeStream.fail_start:
            raise OSError("Device unavailable")
        self.callback(np.full((asr.RATE, 1), 0.1, dtype=np.float32))  # one second of "audio"

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

    def unload(self):
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
    d = object.__new__(asr.Dictation)
    d.np = np
    d.sd = types.SimpleNamespace(InputStream=FakeStream)
    d.model = FakeModel()
    d.lock = threading.Lock()
    d.model_lock = threading.Lock()
    d.last_used = time.monotonic()
    d.turn = threading.Condition()
    d.next_ticket = d.serving = 0
    d.stream = None
    d.indicator = None
    d.hint = None
    d.chunks = []
    d.last_take = None
    d.last_save = 0.0
    d.last_clip_warning = -asr.CLIP_WARN_EVERY
    d.pasted = []
    d.paste = lambda text: d.pasted.append(text.strip())
    d.refresh_hint = lambda: None
    monkeypatch.setattr(asr, "warm_rewrite_model", lambda: None)
    monkeypatch.setattr(asr, "llm_cleanup", lambda text, lang: f"CLEANED:{text}")
    FakeStream.fail_start = False
    return d


def test_overlapping_takes_keep_their_own_mode(monkeypatch):
    """A cleanup take still transcribing must not switch to raw when a plain take starts."""
    d = make_dictation(monkeypatch)
    d.model.release_first.clear()
    d.toggle(rewrite=True)                     # take 1: cleanup
    first = threading.Thread(target=d.toggle)  # stop take 1; it transcribes slowly
    first.start()
    assert d.model.first_started.wait(5)
    d.toggle(rewrite=False)                    # take 2: plain, starts while take 1 transcribes
    second = threading.Thread(target=d.toggle)
    second.start()
    time.sleep(0.2)
    d.model.release_first.set()
    first.join(5)
    second.join(5)
    assert d.pasted == ["CLEANED:take1", "take2"]  # right mode for each, in order


def test_microphone_failure_does_not_wedge(monkeypatch):
    d = make_dictation(monkeypatch)
    FakeStream.fail_start = True
    d.toggle()
    assert d.stream is None
    assert any("microphone" in m.lower() for m in notified)
    FakeStream.fail_start = False
    d.toggle()                                 # the next press records normally
    d.toggle()
    assert d.pasted == ["take1"]


def test_saved_recordings_are_private(monkeypatch):
    d = make_dictation(monkeypatch)
    d.last_take = (np.zeros(asr.RATE, dtype=np.float32), "hello")
    d.save()
    mode = stat.S_IMODE(os.stat(asr.RECORDINGS).st_mode)
    assert mode == 0o700


# --- clipboard ------------------------------------------------------------------------------

def test_clipboard_set_survives_missing_tool(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("wl-copy")
    monkeypatch.setattr(asr.subprocess, "run", missing)
    assert asr.clipboard_set("text") is False


def test_clipboard_set_survives_hang(monkeypatch):
    def hang(*a, **kw):
        raise subprocess.TimeoutExpired("xclip", 2)
    monkeypatch.setattr(asr.subprocess, "run", hang)
    assert asr.clipboard_set("text") is False


# --- memory -------------------------------------------------------------------------------

def test_unload_settings_validate():
    with open(asr.CONFIG, "w") as f:
        f.write('asr_unload_minutes = -5\nllm_unload_minutes = "soon"\n')
    s = asr.settings()
    assert s["asr_unload_minutes"] == 30 and s["llm_unload_minutes"] == 30


@pytest.mark.parametrize("minutes, keep_alive", [(30, "30m"), (0, "0m"), (-1, -1)])
def test_llm_keep_alive(minutes, keep_alive):
    with open(asr.CONFIG, "w") as f:
        f.write(f"llm_unload_minutes = {minutes}\n")
    assert asr.llm_keep_alive() == keep_alive


def test_speech_model_unloads_when_idle_and_reloads_on_use(monkeypatch):
    monkeypatch.setattr(asr, "DEVICE", "cuda")
    with open(asr.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")
    d = make_dictation(monkeypatch)
    d.unload_if_idle()
    assert d.model.loaded()                   # used just now: stays loaded
    d.last_used -= 6 * 60
    d.unload_if_idle()
    assert not d.model.loaded()               # 6 min idle: freed
    d.toggle()                                # the next take reloads it before transcribing
    d.toggle()
    assert d.model.loaded() and d.pasted == ["take1"]


def test_speech_model_kept_when_never_or_cpu(monkeypatch):
    d = make_dictation(monkeypatch)
    d.last_used -= 120 * 60
    monkeypatch.setattr(asr, "DEVICE", "cuda")
    with open(asr.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 0\n")    # never
    d.unload_if_idle()
    assert d.model.loaded()
    monkeypatch.setattr(asr, "DEVICE", "cpu")
    with open(asr.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")    # no GPU memory to free on the CPU
    d.unload_if_idle()
    assert d.model.loaded()


def test_no_unload_while_transcribing(monkeypatch):
    monkeypatch.setattr(asr, "DEVICE", "cuda")
    with open(asr.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")
    d = make_dictation(monkeypatch)
    d.last_used -= 6 * 60
    with d.model_lock:                        # a take is being transcribed
        d.unload_if_idle()
    assert d.model.loaded()
