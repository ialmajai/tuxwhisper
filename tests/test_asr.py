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

real_add_history = asr.add_history  # the fixture below stubs it out for the dictation tests


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every user file at a temp dir and silence desktop side effects."""
    monkeypatch.setattr(asr, "CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setattr(asr, "REPLACEMENTS", str(tmp_path / "replacements.txt"))
    monkeypatch.setattr(asr, "VOCAB", str(tmp_path / "vocab.txt"))
    monkeypatch.setattr(asr, "RECORDINGS", str(tmp_path / "recordings"))
    monkeypatch.setattr(asr, "MODES_FILE", str(tmp_path / "modes.txt"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))  # history, retype job: never the real ones
    monkeypatch.delenv("ASR_MODEL", raising=False)
    monkeypatch.setattr(asr, "sound", lambda name: None)
    monkeypatch.setattr(asr, "close_notification", lambda nid: None)
    monkeypatch.setattr(asr, "add_history", lambda text, **kw: None)
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
    d.fallback_from = None
    d.last_memory_notice = 0.0
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
    d.paste = lambda text: d.pasted.append(text.strip()) or True
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


# --- GPU memory fallback ---------------------------------------------------------------------

class OOMModel(FakeModel):
    """A model parked in RAM that no longer fits back on the GPU."""
    is_loaded = False

    def load(self):
        raise RuntimeError("CUDA failed with error out of memory")


def fake_gpu(monkeypatch, free_gb, loads):
    """Pretend the GPU has free_gb free: bigger models fail to load as out of memory. Models
    that load are appended to loads."""
    monkeypatch.setattr(asr, "DEVICE", "cuda")
    monkeypatch.setattr(asr, "free_gpu_gb", lambda: free_gb[0])
    monkeypatch.setattr(asr, "model_downloaded", lambda name: True)
    def load(name):
        if asr.model_size(name) > free_gb[0]:
            raise RuntimeError("CUDA failed with error out of memory")
        loads.append(name)
        return FakeModel()
    monkeypatch.setattr(asr, "load_whisper", load)


@pytest.mark.parametrize("free, expected, tried", [
    (2.0, "medium", ["medium"]),
    (1.2, "small", ["medium", "small"]),
    (0.3, "base", ["medium", "small", "base"]),
    (0.1, "tiny", ["medium", "small", "base", "tiny"]),
    (0.05, None, ["medium", "small", "base", "tiny"]),
])
def test_load_smaller_tries_each_smaller_model_largest_first(monkeypatch, free, expected, tried):
    fake_gpu(monkeypatch, [free], [])
    names = []
    found = asr.load_smaller("large-v3-turbo", names)
    assert (found[1] if found else None) == expected
    assert names == tried


def test_load_smaller_uses_english_only_models_only_for_english(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [3.0], loads)
    assert asr.load_smaller("large-v3")[1] == "large-v3-turbo"
    asr.save_settings({"language": "en"})
    assert asr.load_smaller("large-v3-turbo")[1] == "distil-large-v3.5"


def test_reload_out_of_memory_falls_back_and_still_pastes(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = OOMModel()
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    d.toggle()
    d.toggle()
    assert loads == ["small"]
    assert d.pasted == ["take1"]
    assert d.fallback_from == "large-v3-turbo"
    assert any("using small" in n for n in notified)


def test_no_model_fits_notifies_once_and_skips_the_take(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = OOMModel()
    fake_gpu(monkeypatch, [0.05], [])
    d.toggle()
    d.toggle()
    assert d.pasted == []
    assert sum("Not enough GPU memory" in n for n in notified) == 1


def test_switches_back_when_memory_frees(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = OOMModel()
    free, loads = [1.2], []
    fake_gpu(monkeypatch, free, loads)
    d.toggle()
    d.toggle()
    free[0] = 4.0
    d.toggle()
    d.toggle()
    assert loads == ["small", asr.asr_model()]
    assert d.fallback_from is None
    assert any("Back to large-v3-turbo" in n for n in notified)


def test_startup_out_of_memory_falls_back(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    _, replaced = asr.load_model()
    assert loads == ["small"] and replaced == "large-v3-turbo"


def test_startup_with_no_room_waits_instead_of_exiting(monkeypatch):
    fake_gpu(monkeypatch, [0.05], [])
    assert asr.load_model() == (None, "large-v3-turbo")
    assert notified == ["Not enough GPU memory to dictate"]


def test_take_loads_a_model_once_there_is_room(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model, d.fallback_from = None, "large-v3-turbo"
    free, loads = [0.05], []
    fake_gpu(monkeypatch, free, loads)
    d.toggle()
    d.toggle()
    assert d.pasted == [] and loads == []
    free[0] = 1.2
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.pasted == ["take1"]


# --- self-corrections ------------------------------------------------------------------------

@pytest.mark.parametrize("text, cued", [
    ("let's meet on tuesday no wait wednesday", True),
    ("send it to john sorry i mean mike", True),
    ("at four actually make that five", True),
    ("call you tomorrow or rather friday", True),
    ("book six scratch that eight", True),
    ("report dot txt no no report dot md", True),
    ("what i mean is that the tests are flaky", False),
    ("is it tuesday or wednesday", False),
    ("there are no errors in the log", False),
    ("i actually liked the first version", False),
])
def test_self_correction_rule_only_for_correction_cues(text, cued):
    assert (asr.SELF_CORRECTION in asr.rewrite_task(text)) is cued
    assert asr.rewrite_task(text).startswith(asr.modes()[asr.current_mode()])


# --- correcting the last transcript ----------------------------------------------------------

def test_suggest_learning_names_get_rules_and_vocabulary():
    got = asr.suggest_learning("I use tux whisper daily.", "I use TuxWhisper daily.")
    assert ("rule", "tux whisper", "TuxWhisper", True) in got
    assert ("vocab", "TuxWhisper", "TuxWhisper", True) in got


def test_suggest_learning_plain_words_start_unticked():
    assert asr.suggest_learning("we need two copies", "we need three copies") == [
        ("rule", "two", "three", False)]


def test_suggest_learning_skips_rewrites_punctuation_and_known_words():
    assert asr.suggest_learning("Hello, world", "Hello world.") == []
    assert asr.suggest_learning("one two three four five six", "a b c d e f") == []
    got = asr.suggest_learning("ask kubernetes", "ask Kubernetes", ["Kubernetes"])
    assert got == [("rule", "kubernetes", "Kubernetes", True)]


def fake_dialogs(monkeypatch, edited, ticked):
    """zenity: the text box returns edited; the checklist returns the ticked row indexes."""
    shown = []
    def dialog(*args):
        shown.append(args)
        if "--text-info" in args:
            return edited
        if "--checklist" in args:
            return " ".join(str(i) for i in ticked)
        return None
    monkeypatch.setattr(asr, "zenity", dialog)
    copied = []
    monkeypatch.setattr(asr, "clipboard_set", lambda text, *a: copied.append(text) or True)
    return shown, copied


def test_correct_last_copies_fix_and_learns_ticked_items(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("I use tux whisper daily.")
    _, copied = fake_dialogs(monkeypatch, "I use TuxWhisper daily.", [0, 1])
    asr.correct_last()
    assert copied == ["I use TuxWhisper daily."]
    assert asr.load_history()[-1]["text"] == "I use TuxWhisper daily."
    assert asr.apply_replacements("tux whisper rocks") == "TuxWhisper rocks"
    assert asr.vocab() == "TuxWhisper"


def test_correct_last_unchanged_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("Fine as it is.")
    shown, copied = fake_dialogs(monkeypatch, "Fine as it is.", [])
    asr.correct_last()
    assert copied == [] and len(shown) == 1


def test_correct_last_fixes_the_saved_recording(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    os.makedirs(asr.RECORDINGS)
    with open(os.path.join(asr.RECORDINGS, "metadata.csv"), "w") as f:
        f.write("file_name,transcription\na.wav,first take\nb.wav,send it to jon\n")
    real_add_history("send it to jon")
    asr.mark_last_saved("b.wav")
    fake_dialogs(monkeypatch, "send it to John", [2])  # 0: "jon" rule, 1: "John" vocabulary, 2: recording
    asr.correct_last()
    with open(os.path.join(asr.RECORDINGS, "metadata.csv")) as f:
        assert f.read().splitlines() == ["file_name,transcription", "a.wav,first take",
                                         "b.wav,send it to John"]


def test_correct_last_never_offers_to_fix_a_rewritten_take(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("Dear team, the build failed.", rewritten=True)
    asr.mark_last_saved("c.wav")
    shown, _ = fake_dialogs(monkeypatch, "Dear team, the build passed.", [])
    asr.correct_last()
    assert not any("saved recording" in a for args in shown for a in args)


class FakeKeyboard:
    def __init__(self):
        self.keys = []

    def write(self, kind, code, val):
        if val:
            self.keys.append(code)

    def syn(self):
        pass


def test_retype_erases_the_dictation_and_pastes_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(asr.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.kbd = FakeKeyboard()
    with open(asr.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert d.retype()
    assert d.kbd.keys == [14] * 6 and d.pasted == ["Hello"]
    assert not os.path.exists(asr.retype_path())


def test_correct_last_replaces_in_place_when_the_daemon_can(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("send it to jon", typed=True)
    jobs = []
    def daemon(cmd):
        with open(asr.retype_path()) as f:
            jobs.append(f.read())
        return "ok"
    monkeypatch.setattr(asr, "request", daemon)
    _, copied = fake_dialogs(monkeypatch, "send it to John", [])
    asr.correct_last()
    assert copied == []  # typed in place, so nothing to paste by hand
    assert jobs == ['{"delete": 15, "text": "send it to John"}']  # 14 characters plus the space
    assert notified[-1] == "✏️ Replaced the dictated text"


def test_typed_length_counts_characters_not_accents():
    assert asr.typed_length("café") == 4
    assert asr.typed_length("café") == 4


# --- robustness ------------------------------------------------------------------------------

def test_settings_with_emoji_and_control_characters_round_trip():
    asr.save_settings({"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"}, "x": "a\x7fb"})
    assert asr.read_config() == {"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"}, "x": "a\x7fb"}


def test_unreadable_settings_are_reported_and_kept(monkeypatch):
    monkeypatch.delattr(asr.read_config, "warned", raising=False)
    with open(asr.CONFIG, "w") as f:
        f.write('mode = "unterminated\n')
    assert asr.read_config() == {}
    assert notified == ["Settings file is unreadable; using defaults"]
    asr.save_settings({"sounds": False})
    kept = [n for n in os.listdir(os.path.dirname(asr.CONFIG)) if ".unreadable-" in n]
    assert len(kept) == 1 and asr.read_config() == {"sounds": False}


def test_failed_take_is_reported_and_the_next_one_works(monkeypatch):
    d = make_dictation(monkeypatch)
    calls = []
    def transcribe(audio, **_):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return [types.SimpleNamespace(text="fine")], None
    d.model.transcribe = transcribe
    d.toggle()
    d.toggle()
    assert "Dictation failed" in notified and d.pasted == []
    d.toggle()
    d.toggle()
    assert d.pasted == ["fine"]


def test_mic_error_on_stop_does_not_wedge(monkeypatch):
    d = make_dictation(monkeypatch)
    d.toggle()
    def broken():
        raise OSError("device unplugged")
    d.stream.stop = broken
    d.toggle()
    assert d.stream is None


def test_list_rows_starting_with_a_dash_are_not_options(monkeypatch):
    cmds = []
    monkeypatch.setattr(asr.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(
        returncode=1, stdout=""))
    for yad in (True, False):
        monkeypatch.setattr(asr, "have_yad", lambda y=yad: y)
        asr.zenity("--list", "--column=i", "--column=Text", "--ok-label=Copy", "0", "- Buy milk")
    for cmd in cmds:
        assert cmd[-3:] == ["--", "0", "- Buy milk"]
        assert not any(a.startswith("--button") for a in cmd[cmd.index("--"):])


@pytest.mark.parametrize("typed, age, expected", [
    (True, 0, True),                          # fresh and typed: replace in place
    (False, 0, False),                        # only copied (paste failed): never erase
    (True, asr.RETYPE_WITHIN + 1, False),     # too old: the cursor has probably moved
])
def test_correct_last_retypes_only_a_fresh_typed_dictation(monkeypatch, typed, age, expected):
    real_add_history("send it to jon", typed=typed)
    items = asr.load_history()
    items[-1]["at"] -= age
    asr.save_history(items)
    asked = []
    monkeypatch.setattr(asr, "request", lambda cmd: asked.append(cmd) or "ok")
    _, copied = fake_dialogs(monkeypatch, "send it to John", [])
    asr.correct_last()
    assert bool(asked) is expected and bool(copied) is not expected


def test_retype_refuses_while_a_take_is_in_flight(monkeypatch):
    monkeypatch.setattr(asr.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.kbd = FakeKeyboard()
    d.next_ticket = 1  # a take stopped but isn't pasted yet
    with open(asr.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert not d.retype() and d.kbd.keys == []


def test_history_records_whether_the_take_was_typed(monkeypatch):
    d = make_dictation(monkeypatch)
    monkeypatch.setattr(asr, "add_history", real_add_history)
    d.paste = lambda text: False  # e.g. the clipboard couldn't be set
    d.toggle()
    d.toggle()
    assert asr.load_history()[-1]["typed"] is False
