"""Tests for settings, history and the files shared by the daemon, the menu and the CLI."""
import os
import subprocess
import sys
import time
import tomllib

import numpy as np
from conftest import make_dictation, notified

from tuxwhisper import store


def test_replacements(tmp_path):
    (tmp_path / "replacements.txt").write_text(
        "# comment => ignored\nnew line => \\n\ncomma => ,\ntux whisper => TuxWhisper\n")
    assert store.apply_replacements("Hello. New line. World.") == "Hello.\nWorld."
    assert store.apply_replacements("I like tux whisper") == "I like TuxWhisper"
    assert store.apply_replacements("commander") == "commander"  # whole words only


def test_deletion_rule_keeps_the_words_around_it_apart(tmp_path):
    (tmp_path / "replacements.txt").write_text("um =>\nnew line => \\n\n")
    assert store.apply_replacements("Hello, um, world.") == "Hello, world."
    assert store.apply_replacements("I um think so.") == "I think so."
    assert store.apply_replacements("Um, I think so.") == "I think so."
    assert store.apply_replacements("Yummy.") == "Yummy."  # whole words only
    assert store.apply_replacements("New line. Hello.") == "\nHello."  # a leading line break stays


def test_replacements_missing_file():
    assert store.apply_replacements("unchanged") == "unchanged"


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
    assert s.sound_volume == 40
    assert s.paste_key == "shift+insert"
    assert s.language == ""  # a wrong-typed language would reach Whisper
    assert s.mode_models == {}


def test_settings_of_the_wrong_type_fall_back_to_their_defaults():
    with open(store.CONFIG, "w") as f:
        f.write('language = ["en"]\npaste_key = ["ctrl+v"]\nsounds = "no"\nstream_long_takes = 0\n'
                'rewrite_model = 5\nasr_model = 3\nmode = 1\nasr_unload_minutes = true\n'
                'mode_models = {"Clean up" = 5, "Email" = "qwen3"}\n')
    s = store.settings()  # a hand-edited file must never stop the daemon or the menu
    assert (s.language, s.paste_key, s.sounds, s.stream_long_takes) == ("", "shift+insert", True, True)
    assert (s.rewrite_model, s.asr_model, s.mode, s.asr_unload_minutes) == ("llama3.2", "", "Clean up", 30)
    assert s.mode_models == {"Email": "qwen3"}


def test_settings_file_that_isn_t_utf8_counts_as_unreadable(monkeypatch):
    monkeypatch.delattr(store.read_config, "warned", raising=False)
    with open(store.CONFIG, "wb") as f:
        f.write(b'mode = "\xff\xfe"\n')
    assert store.read_config() == {} and store.settings().mode == "Clean up"
    assert notified == ["Settings file is unreadable; using defaults"]
    store.save_settings({"sounds": False})  # the unreadable file is kept, not overwritten
    assert any(".unreadable-" in n for n in os.listdir(os.path.dirname(store.CONFIG)))


def test_settings_clamp_volume():
    with open(store.CONFIG, "w") as f:
        f.write("sound_volume = 250\n")
    assert store.settings().sound_volume == 100


def test_bad_env_volume_does_not_crash_import(tmp_path):
    env = dict(os.environ, ASR_SOUND_VOLUME="abc", XDG_CONFIG_HOME=str(tmp_path))
    code = "import tuxwhisper.asr; from tuxwhisper import store; print(store.settings().sound_volume)"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "40"


def test_unload_settings_validate():
    with open(store.CONFIG, "w") as f:
        f.write('asr_unload_minutes = -5\nllm_unload_minutes = "soon"\n')
    s = store.settings()
    assert s.asr_unload_minutes == 30 and s.llm_unload_minutes == 30


def test_settings_with_emoji_and_control_characters_round_trip():
    store.save_settings({"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"}, "rewrite_model": "a\x7fb"})
    assert store.read_config() == {"mode": "🐦 Tweet", "mode_models": {"📧 Email": "qwen3:8b"},
                                   "rewrite_model": "a\x7fb"}


def test_saving_drops_hand_added_keys_that_would_make_the_file_unreadable():
    with open(store.CONFIG, "w") as f:
        f.write('sounds = false\nwhen = 2024-01-01\nbig = inf\n"my key" = 1\n[[t]]\nx = 1\n')
    store.update_settings({"mode": "Formal"})  # saving from the menu
    assert store.read_config() == {"sounds": False, "mode": "Formal"}  # still readable


def test_unreadable_settings_are_reported_and_kept(monkeypatch):
    monkeypatch.delattr(store.read_config, "warned", raising=False)
    with open(store.CONFIG, "w") as f:
        f.write('mode = "unterminated\n')
    assert store.read_config() == {}
    assert notified == ["Settings file is unreadable; using defaults"]
    store.save_settings({"sounds": False})
    kept = [n for n in os.listdir(os.path.dirname(store.CONFIG)) if ".unreadable-" in n]
    assert len(kept) == 1 and store.read_config() == {"sounds": False}


def test_update_settings_keeps_a_change_made_elsewhere_meanwhile():
    store.save_settings({"sounds": True})
    stale = store.read_config()                      # the menu read this...
    store.update_settings({"asr_model": "small"})    # ...then a download switched the model
    stale["sounds"] = False
    store.update_settings({k: v for k, v in stale.items() if v is False})
    assert store.read_config() == {"sounds": False, "asr_model": "small"}


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    path = str(tmp_path / "sub" / "f.txt")
    store.atomic_write(path, "one")
    store.atomic_write(path, "two")
    with open(path) as f:
        assert f.read() == "two"
    assert os.listdir(tmp_path / "sub") == ["f.txt"]


def test_dataset_size_counts_the_listed_recordings_by_file_size(monkeypatch):
    d = make_dictation(monkeypatch)
    for seconds in (1, 2):
        d.recordings.write_recording(np.zeros(seconds * store.RATE, dtype=np.float32), "hi", None)
        time.sleep(1.1)  # recordings are named by the second
    meta = os.path.join(store.RECORDINGS, "metadata.csv")
    assert store.dataset_size(meta) == (2, 3.0)
    first = min(n for n in os.listdir(store.RECORDINGS) if n.endswith(".wav"))
    os.remove(os.path.join(store.RECORDINGS, first))
    assert store.dataset_size(meta) == (1, 2.0)  # deleted by hand


def test_only_one_instance_of_a_dialog_runs(monkeypatch):
    held = []
    monkeypatch.setattr(store, "_instance_locks", held)
    assert store.single_instance("correct")
    assert not store.single_instance("correct")  # a repeat press while it's open
    assert store.single_instance("settings")     # other dialogs are independent
    for f in held:
        f.close()  # what the process exiting does
    assert store.single_instance("correct")
    for f in held:
        f.close()
