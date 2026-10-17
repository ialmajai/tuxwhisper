"""Tests for TuxWhisper's pure helpers and the dictation core, without a mic, GPU or desktop."""
import io
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

from tuxwhisper import daemon, desktop, engine, llm, menu, store

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


# --- keys -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, binding", [
    ("F6", "F6"), ("f12", "F12"), ("Ctrl+Alt+D", "<Control><Alt>d"), ("super+space", "<Super>space"),
    ("Alt+F9", "<Alt>F9"), ("Shift+Insert", "<Shift>Insert"),
    ("d", None), ("space", None), ("Hyper+X", None), ("F25", None), ("Ctrl+", None), ("", None),
])
def test_parse_key(text, binding):
    assert desktop.parse_key(text) == binding


@pytest.mark.parametrize("binding, label", [
    ("<Alt>F9", "Alt+F9"), ("<Control><Alt>d", "Ctrl+Alt+D"), ("<Primary>h", "Ctrl+H"),
    ("F5", "F5"), ("<Super>space", "Super+Space"), ("", ""),
])
def test_key_label(binding, label):
    assert desktop.key_label(binding) == label


def test_key_suggestions_round_trip():
    for label in desktop.KEY_SUGGESTIONS:
        assert desktop.key_label(desktop.parse_key(label)) == label



def test_fit_sizes_list_dialogs():
    rows = zip(["F5", "F3"], ["Dictation", "Cleanup"])  # callers pass iterators
    width, height = desktop.fit(("Key", "Action"), rows)
    assert width.startswith("--width=") and height.startswith("--height=")
    long_row = desktop.fit(("Value",), [("x" * 120,)])[0]
    assert int(long_row.split("=")[1]) > int(width.split("=")[1])  # long cells widen the dialog


def test_yad_args_translate_zenity_options():
    assert desktop.yad_args(("--list", "--ok-label=Copy", "--column=A")) == [
        "--list", "--column=A", "--button=Cancel:1", "--button=Copy:0"]
    assert desktop.yad_args(("--error", "--text=x")) == ["--image=dialog-error", "--text=x", "--button=OK:0"]
    assert desktop.yad_args(("--question", "--text=x"))[-2:] == ["--button=No:1", "--button=Yes:0"]

# --- replacements ---------------------------------------------------------------------------

def test_replacements(tmp_path):
    (tmp_path / "replacements.txt").write_text(
        "# comment => ignored\nnew line => \\n\ncomma => ,\ntux whisper => TuxWhisper\n")
    assert store.apply_replacements("Hello. New line. World.") == "Hello.\nWorld."
    assert store.apply_replacements("I like tux whisper") == "I like TuxWhisper"
    assert store.apply_replacements("commander") == "commander"  # whole words only


def test_replacements_missing_file():
    assert store.apply_replacements("unchanged") == "unchanged"


# --- settings -------------------------------------------------------------------------------

def test_toml_value_round_trips():
    cfg = {"sounds": True, "sound_volume": 57, "language": "", "mode": 'Say "hi"',
           "mode_models": {"Email": "qwen3:8b", 'My "q" mode': "x"}}
    store.save_settings(cfg)
    with open(store.CONFIG, "rb") as f:
        assert tomllib.load(f) == cfg


def test_settings_reject_bad_values():
    with open(store.CONFIG, "w") as f:
        f.write('sound_volume = "loud"\npaste_key = "alt+v"\nlanguage = 7\nmode_models = "x"\n')
    s = store.settings()
    assert s["sound_volume"] == 40
    assert s["paste_key"] == "shift+insert"
    assert s["language"] == ""  # a wrong-typed language would reach Whisper
    assert s["mode_models"] == {}


def test_settings_clamp_volume():
    with open(store.CONFIG, "w") as f:
        f.write("sound_volume = 250\n")
    assert store.settings()["sound_volume"] == 100


def test_bad_env_volume_does_not_crash_import(tmp_path):
    env = dict(os.environ, ASR_SOUND_VOLUME="abc", XDG_CONFIG_HOME=str(tmp_path))
    code = "import tuxwhisper.asr; from tuxwhisper import store; print(store.settings()['sound_volume'])"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
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
    d = daemon.Dictation(FakeModel(), types.SimpleNamespace(InputStream=FakeStream), kbd=None, ec=None)
    d.pasted = []
    d.paste = lambda text: d.pasted.append(text.strip()) or True
    d.refresh_hint = lambda: None
    monkeypatch.setattr(llm, "warm_rewrite_model", lambda: None)
    monkeypatch.setattr(llm, "llm_cleanup", lambda text, lang: f"CLEANED:{text}")
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


class InlineThread:
    """threading.Thread that runs its target on start(), so a test sees the result at once."""
    def __init__(self, target, args=(), daemon=None):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def save_with_answer(monkeypatch, d, answer):
    """Press the save key; the confirmation dialog returns answer (None = Cancel)."""
    shown = []
    monkeypatch.setattr(daemon, "threading", types.SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(desktop, "edit_text", lambda text, title, ok, explain=None: shown.append(text) or answer)
    d.save()
    return shown


def saved_rows():
    with open(os.path.join(store.RECORDINGS, "metadata.csv")) as f:
        return f.read().splitlines()[1:]


def test_saved_recordings_are_private(monkeypatch):
    d = make_dictation(monkeypatch)
    d.last_take = [np.zeros(store.RATE, dtype=np.float32), "hello", None]
    save_with_answer(monkeypatch, d, "hello")
    mode = stat.S_IMODE(os.stat(store.RECORDINGS).st_mode)
    assert mode == 0o700


def test_save_asks_first_and_keeps_the_fixed_transcript(monkeypatch):
    d = make_dictation(monkeypatch)
    d.last_take = [np.zeros(store.RATE, dtype=np.float32), "send it to jon", None]
    shown = save_with_answer(monkeypatch, d, "send it to John")
    assert shown == ["send it to jon"]
    assert [r.split(",", 1)[1] for r in saved_rows()] == ["send it to John"]


def test_save_dialog_that_fails_is_reported_and_the_take_kept(monkeypatch):
    d = make_dictation(monkeypatch)
    take = d.last_take = [np.zeros(store.RATE, dtype=np.float32), "hello", None]
    monkeypatch.setattr(daemon, "threading", types.SimpleNamespace(Thread=InlineThread))
    def missing(*a):
        raise RuntimeError("zenity isn't installed")
    monkeypatch.setattr(desktop, "edit_text", missing)
    d.save()
    assert notified == ["Couldn't open the save dialog"] and d.last_take is take


def test_save_that_fails_to_write_is_reported_and_the_take_kept(monkeypatch):
    d = make_dictation(monkeypatch)
    take = d.last_take = [np.zeros(store.RATE, dtype=np.float32), "hello", None]
    def full(*a):
        raise OSError("No space left on device")
    d.write_recording = full
    save_with_answer(monkeypatch, d, "hello")
    assert notified == ["Couldn't save the recording"] and d.last_take is take


@pytest.mark.parametrize("err, body", [
    (OSError(28, "No space left on device"), "The disk is full."),
    (PermissionError(13, "Permission denied"), "No permission to write to {}."),
    (ValueError("odd"), "odd")])
def test_save_errors_are_said_plainly(err, body):
    assert daemon.write_error(err) == body.format(store.RECORDINGS)


def test_cancelled_save_does_not_bring_back_an_older_take(monkeypatch):
    d = make_dictation(monkeypatch)
    d.last_take = [np.zeros(store.RATE, dtype=np.float32), "old", None]
    monkeypatch.setattr(daemon, "threading", types.SimpleNamespace(Thread=InlineThread))
    def dialog(text, title, ok, explain=None):
        # While it's open, a newer take is dictated and the save key takes it for its own dialog.
        d.take_id += 1
        d.last_take = None
        # ... and then this one is cancelled (returns None)
    monkeypatch.setattr(desktop, "edit_text", dialog)
    d.save()
    assert d.last_take is None


@pytest.mark.parametrize("code, result", [(0, "text"), (1, None), (252, None), (5, RuntimeError)])
def test_dialog_failure_is_not_mistaken_for_cancel(monkeypatch, code, result):
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=code, stdout="text\n", stderr="cannot open display"))
    if result is RuntimeError:
        with pytest.raises(RuntimeError):
            desktop.zenity("--entry")
    else:
        assert desktop.zenity("--entry") == result


def test_missing_dialog_tool_raises(monkeypatch):
    def missing(cmd, **k):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(desktop.subprocess, "run", missing)
    with pytest.raises(RuntimeError):
        desktop.zenity("--entry")


def test_edit_text_drafts_are_private_unique_and_removed(monkeypatch, tmp_path):
    seen = []
    def dialog(*args):
        path = next(a for a in args if a.startswith("--filename=")).removeprefix("--filename=")
        seen.append(path)
        os.remove(path)  # already gone (e.g. the session ended) must not raise
        return "x"
    monkeypatch.setattr(desktop, "zenity", dialog)
    assert desktop.edit_text("a", "t", "OK") == "x" and desktop.edit_text("a", "t", "OK") == "x"
    assert seen[0] != seen[1] and all(os.path.dirname(p) == str(tmp_path) for p in seen)
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    with pytest.raises(OSError):
        desktop.edit_text("a", "t", "OK")


def test_cancelled_save_writes_nothing_and_can_be_asked_again(monkeypatch):
    d = make_dictation(monkeypatch)
    take = [np.zeros(store.RATE, dtype=np.float32), "hello", None]
    d.last_take = take
    save_with_answer(monkeypatch, d, None)
    assert not os.path.exists(store.RECORDINGS)
    assert d.last_take is take  # the save key offers it again


# --- clipboard ------------------------------------------------------------------------------

def test_clipboard_set_survives_missing_tool(monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("wl-copy")
    monkeypatch.setattr(desktop.subprocess, "run", missing)
    assert desktop.clipboard_set("text") is False


def test_clipboard_set_survives_hang(monkeypatch):
    def hang(*a, **kw):
        raise subprocess.TimeoutExpired("xclip", 2)
    monkeypatch.setattr(desktop.subprocess, "run", hang)
    assert desktop.clipboard_set("text") is False


# --- memory -------------------------------------------------------------------------------

def test_unload_settings_validate():
    with open(store.CONFIG, "w") as f:
        f.write('asr_unload_minutes = -5\nllm_unload_minutes = "soon"\n')
    s = store.settings()
    assert s["asr_unload_minutes"] == 30 and s["llm_unload_minutes"] == 30


@pytest.mark.parametrize("minutes, keep_alive", [(30, "30m"), (0, "0m"), (-1, -1)])
def test_llm_keep_alive(minutes, keep_alive):
    with open(store.CONFIG, "w") as f:
        f.write(f"llm_unload_minutes = {minutes}\n")
    assert llm.llm_keep_alive() == keep_alive


def test_speech_model_unloads_when_idle_and_reloads_on_use(monkeypatch):
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
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
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 0\n")    # never
    d.unload_if_idle()
    assert d.model.loaded()
    monkeypatch.setattr(engine, "DEVICE", "cpu")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")    # no GPU memory to free on the CPU
    d.unload_if_idle()
    assert d.model.loaded()


def test_no_unload_while_transcribing(monkeypatch):
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
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
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    monkeypatch.setattr(engine, "free_gpu_gb", lambda: free_gb[0])
    monkeypatch.setattr(engine, "model_downloaded", lambda name: True)
    def load(name):
        if engine.model_size(name) > free_gb[0]:
            raise RuntimeError("CUDA failed with error out of memory")
        loads.append(name)
        return FakeModel()
    monkeypatch.setattr(engine, "load_whisper", load)


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
    found = engine.load_smaller("large-v3-turbo", names)
    assert (found[1] if found else None) == expected
    assert names == tried


def test_load_smaller_uses_english_only_models_only_for_english(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [3.0], loads)
    assert engine.load_smaller("large-v3")[1] == "large-v3-turbo"
    store.save_settings({"language": "en"})
    assert engine.load_smaller("large-v3-turbo")[1] == "distil-large-v3.5"


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
    d.restore_preferred_model()  # the idle thread does this between takes
    assert loads == ["small", engine.asr_model()]
    assert d.fallback_from is None
    assert any("Back to large-v3-turbo" in n for n in notified)


def test_startup_out_of_memory_falls_back(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    _, name, replaced = engine.load_model()
    assert loads == ["small"] and name == "small" and replaced == "large-v3-turbo"


def test_startup_with_no_room_waits_instead_of_exiting(monkeypatch):
    fake_gpu(monkeypatch, [0.05], [])
    assert engine.load_model() == (None, None, "large-v3-turbo")
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
    assert (llm.SELF_CORRECTION in llm.rewrite_task(text)) is cued
    assert llm.rewrite_task(text).startswith(llm.modes()[llm.current_mode()])


# --- correcting the last transcript ----------------------------------------------------------

def test_suggest_learning_names_get_rules_and_vocabulary():
    got = menu.suggest_learning("I use tux whisper daily.", "I use TuxWhisper daily.")
    assert ("rule", "tux whisper", "TuxWhisper", True) in got
    assert ("vocab", "TuxWhisper", "TuxWhisper", True) in got


def test_suggest_learning_plain_words_start_unticked():
    assert menu.suggest_learning("we need two copies", "we need three copies") == [
        ("rule", "two", "three", False)]


def test_suggest_learning_skips_rewrites_punctuation_and_known_words():
    assert menu.suggest_learning("Hello, world", "Hello world.") == []
    assert menu.suggest_learning("one two three four five six", "a b c d e f") == []
    got = menu.suggest_learning("ask kubernetes", "ask Kubernetes", ["Kubernetes"])
    assert got == [("rule", "kubernetes", "Kubernetes", True)]


def fake_dialogs(monkeypatch, edited, ticked):
    """zenity: the text box returns edited; the checklist ticks the rows whose label contains
    one of the ticked strings, and returns their indexes."""
    shown = []
    def dialog(*args):
        shown.append(args)
        if "--text-info" in args:
            return edited
        if "--checklist" in args:  # rows (tick, index, label) follow the options
            rows = args[next(i for i, a in enumerate(args) if not a.startswith("-")):]
            return " ".join(i for i, label in zip(rows[1::3], rows[2::3]) if any(t in label for t in ticked))
        return None
    monkeypatch.setattr(desktop, "zenity", dialog)
    copied = []
    monkeypatch.setattr(desktop, "clipboard_set", lambda text, *a: copied.append(text) or True)
    return shown, copied


def test_correct_last_copies_fix_and_learns_ticked_items(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("I use tux whisper daily.")
    _, copied = fake_dialogs(monkeypatch, "I use TuxWhisper daily.",
                             ['"tux whisper" as "TuxWhisper"', 'Add "TuxWhisper" to the vocabulary'])
    menu.correct_last()
    assert copied == ["I use TuxWhisper daily."]
    assert store.load_history()[-1]["text"] == "I use TuxWhisper daily."
    assert store.apply_replacements("tux whisper rocks") == "TuxWhisper rocks"
    assert store.vocab() == "TuxWhisper"


def test_correct_last_unchanged_does_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("Fine as it is.")
    shown, copied = fake_dialogs(monkeypatch, "Fine as it is.", [])
    menu.correct_last()
    assert copied == [] and len(shown) == 1


def test_correct_last_fixes_the_saved_recording(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    os.makedirs(store.RECORDINGS)
    with open(os.path.join(store.RECORDINGS, "metadata.csv"), "w") as f:
        f.write("file_name,transcription\na.wav,first take\nb.wav,send it to jon\n")
    real_add_history("send it to jon")
    store.mark_last_saved("b.wav")
    fake_dialogs(monkeypatch, "send it to John", ["saved recording"])  # not the "jon" rule or vocabulary
    menu.correct_last()
    with open(os.path.join(store.RECORDINGS, "metadata.csv")) as f:
        assert f.read().splitlines() == ["file_name,transcription", "a.wav,first take",
                                         "b.wav,send it to John"]


def test_correct_last_never_offers_to_fix_a_rewritten_take(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("Dear team, the build failed.", rewritten=True)
    store.mark_last_saved("c.wav")
    shown, _ = fake_dialogs(monkeypatch, "Dear team, the build passed.", [])
    menu.correct_last()
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
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.kbd = FakeKeyboard()
    with open(store.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert d.retype()
    assert d.kbd.keys == [14] * 6 and d.pasted == ["Hello"]
    assert not os.path.exists(store.retype_path())


def test_correct_last_replaces_in_place_when_the_daemon_can(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("send it to jon", typed=True)
    jobs = []
    def fake_request(cmd):
        with open(store.retype_path()) as f:
            jobs.append(f.read())
        return "ok"
    monkeypatch.setattr(daemon, "request", fake_request)
    _, copied = fake_dialogs(monkeypatch, "send it to John", [])
    menu.correct_last()
    assert copied == []  # typed in place, so nothing to paste by hand
    assert jobs == ['{"delete": 15, "text": "send it to John"}']  # 14 characters plus the space
    assert notified[-1] == "Replaced the dictated text"


def test_typed_length_counts_characters_not_accents():
    assert daemon.typed_length("café") == 4
    assert daemon.typed_length("café") == 4


# --- robustness ------------------------------------------------------------------------------

def test_settings_with_emoji_and_control_characters_round_trip():
    store.save_settings({"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"}, "x": "a\x7fb"})
    assert store.read_config() == {"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"}, "x": "a\x7fb"}


def test_unreadable_settings_are_reported_and_kept(monkeypatch):
    monkeypatch.delattr(store.read_config, "warned", raising=False)
    with open(store.CONFIG, "w") as f:
        f.write('mode = "unterminated\n')
    assert store.read_config() == {}
    assert notified == ["Settings file is unreadable; using defaults"]
    store.save_settings({"sounds": False})
    kept = [n for n in os.listdir(os.path.dirname(store.CONFIG)) if ".unreadable-" in n]
    assert len(kept) == 1 and store.read_config() == {"sounds": False}


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
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(
        returncode=1, stdout=""))
    for yad in (True, False):
        monkeypatch.setattr(desktop, "have_yad", lambda y=yad: y)
        desktop.zenity("--list", "--column=i", "--column=Text", "--ok-label=Copy", "0", "- Buy milk")
    for cmd in cmds:
        assert cmd[-3:] == ["--", "0", "- Buy milk"]
        assert not any(a.startswith("--button") for a in cmd[cmd.index("--"):])


@pytest.mark.parametrize("typed, age, expected", [
    (True, 0, True),                          # fresh and typed: replace in place
    (False, 0, False),                        # only copied (paste failed): never erase
    (True, menu.RETYPE_WITHIN + 1, False),     # too old: the cursor has probably moved
])
def test_correct_last_retypes_only_a_fresh_typed_dictation(monkeypatch, typed, age, expected):
    real_add_history("send it to jon", typed=typed)
    items = store.load_history()
    items[-1]["at"] -= age
    store.save_history(items)
    asked = []
    monkeypatch.setattr(daemon, "request", lambda cmd: asked.append(cmd) or "ok")
    _, copied = fake_dialogs(monkeypatch, "send it to John", [])
    menu.correct_last()
    assert bool(asked) is expected and bool(copied) is not expected


@pytest.mark.parametrize("typed, age, label", [
    (True, 0, "Replace"), (False, 0, "Copy"), (True, menu.RETYPE_WITHIN + 1, "Copy")])
def test_correct_last_button_says_what_it_will_do(monkeypatch, typed, age, label):
    real_add_history("send it to jon", typed=typed)
    items = store.load_history()
    items[-1]["at"] -= age
    store.save_history(items)
    monkeypatch.setattr(daemon, "request", lambda cmd: "ok")
    shown, _ = fake_dialogs(monkeypatch, "send it to jon", [])
    menu.correct_last()
    assert f"--ok-label={label}" in shown[0]


def test_retype_refuses_while_a_take_is_in_flight(monkeypatch):
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.kbd = FakeKeyboard()
    d.next_ticket = 1  # a take stopped but isn't pasted yet
    with open(store.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert not d.retype() and d.kbd.keys == []


def test_history_records_whether_the_take_was_typed(monkeypatch):
    d = make_dictation(monkeypatch)
    monkeypatch.setattr(store, "add_history", real_add_history)
    d.paste = lambda text: False  # e.g. the clipboard couldn't be set
    d.toggle()
    d.toggle()
    assert store.load_history()[-1]["typed"] is False


# --- files shared by the daemon and the menu -------------------------------------------------

def test_update_settings_keeps_a_change_made_elsewhere_meanwhile():
    store.save_settings({"sounds": True})
    stale = store.read_config()                      # the menu read this...
    store.update_settings({"asr_model": "small"})    # ...then a download switched the model
    stale["sounds"] = False
    store.update_settings({k: v for k, v in stale.items() if v is False})
    assert store.read_config() == {"sounds": False, "asr_model": "small"}


def test_correction_keeps_takes_dictated_while_its_dialog_was_open(monkeypatch):
    real_add_history("send it to jon")
    fake_dialogs(monkeypatch, "send it to John", [])
    dialog = desktop.zenity
    def slow_dialog(*args):
        if "--text-info" in args:
            real_add_history("a new take")  # the daemon pastes another take meanwhile
        return dialog(*args)
    monkeypatch.setattr(desktop, "zenity", slow_dialog)
    menu.correct_last()
    assert [h["text"] for h in store.load_history()] == ["send it to John", "a new take"]


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    path = str(tmp_path / "sub" / "f.txt")
    store.atomic_write(path, "one")
    store.atomic_write(path, "two")
    with open(path) as f:
        assert f.read() == "two"
    assert os.listdir(tmp_path / "sub") == ["f.txt"]


# --- push-to-talk timing and the take's hot path ---------------------------------------------

@pytest.mark.parametrize("out, window", [("uint32 500\n", 0.7), ("uint32 900\n", 1.1), ("", 0.7)])
def test_repeat_window_follows_the_desktop_repeat_delay(monkeypatch, out, window):
    monkeypatch.setattr(daemon.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert daemon.repeat_window() == pytest.approx(window)


def test_stop_before_recording_notice_closes_it(monkeypatch):
    d = make_dictation(monkeypatch)
    closed = []
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    def slow_notify(msg, **kw):
        if msg == "Recording…":
            d.toggle()  # the take stops while the notice is still being shown
            return "42"
    monkeypatch.setattr(desktop, "notify", slow_notify)
    d.toggle()
    assert "42" in closed and d.indicator is None


def test_recording_notice_becomes_transcribing_and_is_closed(monkeypatch):
    d = make_dictation(monkeypatch)
    shown, closed = [], []
    def fake_notify(msg, **kw):
        shown.append((msg, kw.get("replace")))
        return "7" if kw.get("persistent") else None
    monkeypatch.setattr(desktop, "notify", fake_notify)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    d.toggle()
    d.toggle()
    assert shown[:2] == [("Recording…", None), ("Transcribing…", "7")]  # replaced in place
    assert closed == ["7"] and d.pasted


def test_transcribing_notice_is_closed_when_a_take_fails(monkeypatch):
    d = make_dictation(monkeypatch)
    closed = []
    monkeypatch.setattr(desktop, "notify", lambda msg, **kw: "7" if kw.get("persistent") else None)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    def boom(*a, **k):
        raise RuntimeError("boom")
    d.model.transcribe = boom
    d.toggle()
    d.toggle()
    assert closed == ["7"]


@pytest.mark.parametrize("level, body, timeout", [
    ("info", None, "1500"), ("info", "b", "6000"), ("warn", None, "8000"), ("error", "b", "8000")])
def test_notify_timeout_follows_its_level(monkeypatch, level, body, timeout):
    cmds = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda cmd, **k: cmds.append(cmd))
    real_notify("Title", body=body, level=level)
    assert cmds[0][cmds[0].index("-t") + 1] == timeout


def test_persistent_notify_stays_and_can_replace(monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(
        stdout="9\n"))
    assert real_notify("Transcribing…", persistent=True, icon="mic", replace="5") == "9"
    cmd = cmds[0]
    assert cmd[cmd.index("-t") + 1] == "0" and cmd[cmd.index("-u") + 1] == "normal"
    assert cmd[cmd.index("-r") + 1] == "5" and cmd[cmd.index("-i") + 1] == "mic"


def test_slow_cleanup_replaces_transcribing_in_place(monkeypatch):
    d = make_dictation(monkeypatch)
    shown, closed = [], []
    def fake_notify(msg, **kw):
        shown.append((msg, kw.get("replace")))
        return "7" if kw.get("persistent") else None
    monkeypatch.setattr(desktop, "notify", fake_notify)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    monkeypatch.setattr(daemon.threading, "Timer", lambda delay, fn: types.SimpleNamespace(
        start=fn, cancel=lambda: None, join=lambda: None))  # the cleanup is slow: fires at once
    d.toggle(rewrite=True)
    d.toggle()
    assert [m for m in shown if m[0].startswith("Cleaning up")] == [(f"Cleaning up ({llm.current_mode()})…", "7")]
    assert closed == ["7"]


def test_plural():
    assert desktop.plural(1, "word", "words") == "1 word"
    assert desktop.plural(0, "word", "words") == "0 words"
    assert desktop.plural(3, "recording", "recordings") == "3 recordings"


def test_paste_key_labels_map_to_stored_values(monkeypatch):
    assert set(menu.PASTE_LABELS) == set(store.PASTE_KEYS)
    picks = iter(["Ctrl+Shift+V"])
    seen = []
    def fake_pick(title, options, current):
        seen.append((options, current))
        return next(picks)
    choices = iter(["paste_key", None])
    monkeypatch.setattr(menu, "menu_choice", lambda rows: next(choices))
    monkeypatch.setattr(menu, "keys_summary", lambda: "")
    monkeypatch.setattr(menu.shutil, "which", lambda name: "/usr/bin/zenity")
    monkeypatch.setattr(desktop, "pick", fake_pick)
    menu.settings_menu()
    assert seen == [(["Shift+Insert", "Ctrl+V", "Ctrl+Shift+V"], "Shift+Insert")]
    assert store.read_config()["paste_key"] == "ctrl+shift+v"


def test_no_model_switch_while_recording(monkeypatch):
    d = make_dictation(monkeypatch)
    d.fallback_from = "large-v3-turbo"
    loads = []
    fake_gpu(monkeypatch, [4.0], loads)
    d.stream = object()
    d.restore_preferred_model()
    assert loads == []


# --- shortcuts and downloads -----------------------------------------------------------------

@pytest.mark.parametrize("command, ours", [
    ("/home/me/tuxwhisper/asr toggle", True),
    ("'/home/me/my apps/asr' toggle", True),
    ("/home/me/.local/bin/tuxwhisper toggle", True),
    ("/opt/fooasr toggle", False),           # another program whose name ends in "asr"
    ("notify-send asr toggle", False),
    ("/home/me/tuxwhisper/asr toggle --x", False),
    ("/home/me/tuxwhisper/asr save", False),
    ("'unbalanced", False),
])
def test_runs_command_matches_only_our_program(command, ours):
    assert desktop.runs_command(command, "toggle") is ours


def test_settings_shortcut_label_matches_the_keys_menu(monkeypatch):
    monkeypatch.setattr(desktop, "gnome_shortcuts", lambda: {"/p/": ("<Control><Alt>d", "/x/asr settings")})
    assert desktop.settings_shortcut() == "Ctrl+Alt+D"


def test_download_verified_refuses_a_wrong_checksum(monkeypatch, tmp_path):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass
    monkeypatch.setattr(engine.urllib.request, "urlopen", lambda url, timeout: Resp(b"model"))
    good = engine.hashlib.sha256(b"model").hexdigest()
    engine.download_verified("http://x", str(tmp_path / "ok.bin"), good)
    assert (tmp_path / "ok.bin").read_bytes() == b"model"
    with pytest.raises(OSError):
        engine.download_verified("http://x", str(tmp_path / "bad.bin"), "0" * 64)
    assert not (tmp_path / "bad.bin").exists()


# --- takes that lose words -------------------------------------------------------------------

@pytest.mark.parametrize("seconds, text, title", [
    (0.5, "", None),                                     # a stray key press
    (2.7, "", "Didn't catch that"),
    (20.8, "How would it bother how would?", "Only caught a few words"),
    (20.0, " ".join(["word"] * 40), None),               # normal speech with pauses
    (6.0, "Yes.", None),                                 # short takes can be short
])
def test_missed_speech(seconds, text, title):
    missed = daemon.missed_speech(seconds, text)
    assert (missed[0] if missed else None) == title


def test_empty_take_is_reported(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model.transcribe = lambda audio, **_: ([], None)
    d.toggle()
    d.toggle()
    assert "Didn't catch that" in notified and d.pasted == []


def test_context_size_grows_with_the_transcript():
    assert llm.context_size("x" * 1500, "word " * 50) == 1024
    assert llm.context_size("x" * 1500, "word " * 700) == 4096
    assert llm.context_size("x" * 1500, "word " * 1100) == 8192
    assert llm.context_size("x" * 1500, "word " * 3000) is None


@pytest.mark.parametrize("reply", [
    {"done_reason": "length", "prompt_eval_count": 10, "eval_count": 10},
    {"done_reason": "stop", "prompt_eval_count": 1000, "eval_count": 30},  # input was trimmed
])
def test_cleanup_that_ran_out_of_room_pastes_the_plain_transcript(monkeypatch, reply):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda path, payload, timeout=30: {
        **reply, "message": {"content": '{"cleaned": "half of it"}'}})
    assert llm.llm_cleanup("um the whole take", "en") == "um the whole take"
    assert notified == ["Too long for LLM cleanup; pasted the plain transcript"]


def test_cleanup_without_token_counts_is_kept(monkeypatch):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda path, payload, timeout=30: {
        "done_reason": "stop", "prompt_eval_count": 0, "message": {"content": '{"cleaned": "The take."}'}})
    assert llm.llm_cleanup("um the take", "en") == "The take."


def test_far_too_long_take_skips_cleanup(monkeypatch):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda *a, **k: pytest.fail("should not be called"))
    text = "word " * 5000
    assert llm.llm_cleanup(text, "en") == text


# --- whispered words -------------------------------------------------------------------------

def burst(seconds, total=4.0, level=0.01, rate=16000):
    """A take of total seconds: faint noise with seconds of louder sound in the middle."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.0002, int(total * rate)).astype(np.float32)
    start = int((total - seconds) / 2 * rate)
    a[start:start + int(seconds * rate)] += rng.normal(0, level, int(seconds * rate)).astype(np.float32)
    return a


def test_sound_seconds_counts_sound_not_the_whole_take():
    assert engine.sound_seconds(np.zeros(4 * 16000, dtype=np.float32)) == 0.0
    assert engine.sound_seconds(burst(0.5)) == pytest.approx(0.5, abs=0.06)


def test_sound_seconds_counts_a_take_that_is_sound_throughout():
    assert engine.sound_seconds(burst(4.0)) == pytest.approx(4.0, abs=0.06)


def test_sound_seconds_ignores_a_click_in_digital_silence():
    a = np.zeros(4 * 16000, dtype=np.float32)
    a[30000:30010] = 0.01
    assert engine.sound_seconds(a) <= 0.02  # the click's own frame at most, not the whole take


def test_syllables_count_each_cjk_character():
    assert engine.syllables("你好") == 2
    assert engine.syllables("ありがとう") == 5
    assert engine.syllables("안녕하세요") == 5
    assert engine.syllables("Да") == 1  # other scripts: about one per 2.5 letters


@pytest.mark.parametrize("text, sound, ok", [
    ("Yes.", 0.3, True),
    ("Thank you.", 0.3, True),                 # 6.7 syllables a second: fast but possible
    ("God bless you.", 0.3, False),            # invented from a whispered "yes": 10 a second
    ("This is a sentence, with punctuation and capitals.", 2.0, False),  # the prompt echoed
    (". . . . .", 1.0, False),
])
def test_plausible(text, sound, ok):
    assert engine.plausible(text, burst(sound)) is ok


class WhisperedModel(FakeModel):
    """Like Whisper on a whispered take: the voice filter finds nothing; without it, `heard`."""
    def __init__(self, heard):
        super().__init__()
        self.heard = heard

    def transcribe(self, audio, vad_filter=True, **_):
        return ([] if vad_filter else [types.SimpleNamespace(text=self.heard)]), None


@pytest.mark.parametrize("heard, sound, typed", [
    ("Yes.", 0.4, ["Yes."]),
    ("God bless you.", 0.3, []),
])
def test_whispered_take_is_retried_without_the_voice_filter(monkeypatch, heard, sound, typed):
    d = make_dictation(monkeypatch)
    d.model = WhisperedModel(heard)
    d.process([burst(sound)[:, None]], 0, False, None)
    assert d.pasted == typed
    assert ("Didn't catch that" in notified) is (not typed)


def test_silence_is_not_retried(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = WhisperedModel("You")
    d.process([np.zeros((4 * 16000, 1), dtype=np.float32)], 0, False, None)
    assert d.pasted == [] and "Didn't catch that" in notified


# --- the GPU fills up after loading; the daemon is down --------------------------------------

class RunOOMModel(FakeModel):
    """Loaded, but running it fails: another program took the GPU memory it needs."""
    def detect_language(self, audio, **_):
        raise RuntimeError("CUDA failed with error out of memory")

    def transcribe(self, audio, **_):
        raise RuntimeError("CUDA failed with error out of memory")


def test_take_that_runs_out_of_gpu_memory_switches_down_and_still_pastes(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = RunOOMModel()
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.pasted == ["take1"]
    assert any("using small" in n for n in notified)


def fake_systemctl(monkeypatch, state):
    """The daemon isn't answering and systemd reports it as state; the commands run are returned."""
    monkeypatch.setattr(daemon, "request", lambda cmd, timeout=5: None)
    ran = []
    monkeypatch.setattr(daemon.subprocess, "run", lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(
        returncode=0, stdout=state + "\n"))
    with pytest.raises(SystemExit):
        daemon.send("toggle")
    return ran


@pytest.mark.parametrize("state", ["failed", "inactive"])
def test_key_starts_the_daemon_when_it_is_not_running(monkeypatch, state):
    assert fake_systemctl(monkeypatch, state)[1:] == [["systemctl", "--user", "reset-failed", "tuxwhisper"],
                                                      ["systemctl", "--user", "start", "--no-block", "tuxwhisper"]]
    assert notified == ["Starting TuxWhisper…"]


@pytest.mark.parametrize("state", ["activating", "active"])
def test_key_does_not_restart_a_daemon_still_loading(monkeypatch, state):
    assert fake_systemctl(monkeypatch, state) == [["systemctl", "--user", "is-active", "tuxwhisper"]]
    assert notified == ["TuxWhisper is still starting"]


def test_take_that_runs_out_of_memory_again_after_switching_down_is_reported(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = RunOOMModel()
    fake_gpu(monkeypatch, [1.2], [])
    monkeypatch.setattr(engine, "load_whisper", lambda name: RunOOMModel())  # loads, then can't run
    d.toggle()
    d.toggle()
    assert d.pasted == [] and notified[-1] == "Not enough GPU memory to dictate"


def test_take_with_no_model_that_fits_shows_only_the_memory_notice(monkeypatch):
    d = make_dictation(monkeypatch)
    d.model = RunOOMModel()
    fake_gpu(monkeypatch, [0.05], [])
    d.toggle()
    d.toggle()
    assert "Not enough GPU memory to dictate" in notified and "Didn't catch that" not in notified


def test_out_of_memory_steps_down_from_the_model_in_use(monkeypatch):
    d = make_dictation(monkeypatch)
    loads = []
    fake_gpu(monkeypatch, [1.6], loads)
    d.model, d.model_name, d.fallback_from = RunOOMModel(), "medium", "large-v3-turbo"
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.model_name == "small"  # medium would fit, but just failed
    assert d.fallback_from == "large-v3-turbo"  # still back to the chosen model once there's room


def test_failed_warm_up_frees_the_model(monkeypatch):
    freed = []

    class Broken:
        def transcribe(self, audio):
            raise RuntimeError("CUDA failed with error out of memory")

        def unload(self):
            freed.append(True)

    monkeypatch.setattr(engine, "DEVICE", "vulkan")
    monkeypatch.setattr(engine, "WhisperCpp", lambda name: Broken())
    with pytest.raises(RuntimeError):
        engine.load_whisper("tiny")
    assert freed == [True]
