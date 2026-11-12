"""Tests for the settings menu and correcting the last transcript."""
import json
import os
import types

import pytest
from conftest import notified, real_add_history

from tuxwhisper import client, desktop, menu, store


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


@pytest.mark.parametrize("changed", ["rewritten", "replaced"])
def test_correct_last_never_offers_to_fix_a_rewritten_or_replaced_take(monkeypatch, tmp_path, changed):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    real_add_history("Dear team, the build failed.", **{changed: True})
    store.mark_last_saved("c.wav")
    shown, _ = fake_dialogs(monkeypatch, "Dear team, the build passed.", [])
    menu.correct_last()
    assert not any("saved recording" in a for args in shown for a in args)


def test_correct_last_replaces_in_place_when_the_daemon_can(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    at = real_add_history("send it to jon", typed=True)["at"]
    jobs = []
    def fake_request(cmd):
        with open(store.retype_path()) as f:
            jobs.append(f.read())
        return "ok"
    monkeypatch.setattr(client, "request", fake_request)
    _, copied = fake_dialogs(monkeypatch, "send it to John", [])
    menu.correct_last()
    assert copied == []  # typed in place, so nothing to paste by hand
    assert [json.loads(j) for j in jobs] == [{"delete": 15, "text": "send it to John", "at": at}]  # 14 + the space
    assert notified[-1] == "Replaced the dictated text"


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
    monkeypatch.setattr(client, "request", lambda cmd: asked.append(cmd) or "ok")
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
    monkeypatch.setattr(client, "request", lambda cmd: "ok")
    shown, _ = fake_dialogs(monkeypatch, "send it to jon", [])
    menu.correct_last()
    assert f"--ok-label={label}" in shown[0]


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


def drive_settings_menu(monkeypatch, picks, answers):
    """Run settings_menu with fake dialogs: the menu picks each of picks in turn, then Close;
    every other dialog returns the next of answers. Returns the menu rows shown each time,
    as {key: value}."""
    picks, answers, shown = iter(picks), iter(answers), []
    def dialog(*args):
        if "--title=TuxWhisper Settings" in args:
            rows = args[next(i for i, a in enumerate(args) if not a.startswith("-")):]
            shown.append(dict(zip(rows[0::4], rows[2::4])))
            return next(picks, None)
        return next(answers)
    monkeypatch.setattr(desktop, "have_yad", lambda: False)
    monkeypatch.setattr(desktop, "zenity", dialog)
    monkeypatch.setattr(desktop, "gnome_shortcuts", lambda: None)
    monkeypatch.setattr(menu.shutil, "which", lambda name: "/usr/bin/zenity")
    menu.settings_menu()
    return shown


def test_settings_menu_toggles_sounds(monkeypatch):
    shown = drive_settings_menu(monkeypatch, ["sounds"], [])
    assert store.read_config() == {"sounds": False}
    assert shown[0]["sounds"] == "On (default)" and shown[1]["sounds"] == "Off"


def test_settings_menu_sets_the_language(monkeypatch):
    shown = drive_settings_menu(monkeypatch, ["language"], ["French (fr)"])
    assert store.read_config() == {"language": "fr"}
    assert shown[0]["language"] == "auto-detect (default)" and shown[1]["language"] == "French"


def test_settings_menu_sets_the_cleanup_style(monkeypatch):
    shown = drive_settings_menu(monkeypatch, ["mode"], ["Email"])
    assert store.read_config() == {"mode": "Email"}
    assert shown[1]["mode"] == "Email"


def test_settings_menu_sets_the_paste_key(monkeypatch):
    drive_settings_menu(monkeypatch, ["paste_key"], ["Ctrl+V"])
    assert store.read_config() == {"paste_key": "ctrl+v"}


def test_settings_menu_saves_the_volume_then_previews_it(monkeypatch):
    played = []
    monkeypatch.setattr(desktop, "sound", lambda name: played.append(store.settings().sound_volume))
    drive_settings_menu(monkeypatch, ["sound_volume"], ["55"])
    assert store.read_config() == {"sound_volume": 55} and played == [55]


def test_settings_menu_cancelled_dialog_saves_nothing(monkeypatch):
    drive_settings_menu(monkeypatch, ["language", "paste_key", "sound_volume"], [None, None, None])
    assert not os.path.exists(store.CONFIG)


def test_menu_correction_waits_for_one_opened_with_the_key(monkeypatch):
    opened = []
    monkeypatch.setattr(menu, "correct_last", lambda: opened.append(True))
    monkeypatch.setattr(store, "_instance_locks", [])
    assert store.single_instance("correct")  # F6's correction window is open
    menu.correct_last_once()
    assert opened == [] and notified[-1] == "A correction is already open"
    for f in store._instance_locks:
        f.close()  # F6's window closed
    menu.correct_last_once()
    menu.correct_last_once()  # the menu's own lock was let go after the first
    assert opened == [True, True]


@pytest.mark.parametrize("picked, typed, saved", [
    ("open for 10 min after a take", None, 10),
    ("Other…", "45", 45),                       # any number of minutes
    ("Other…", " 0 ", 0),                       # released after each take
    ("Other…", "soon", None),                   # not a number: nothing changes
    ("Other…", "99999", None),                  # more than a day
    ("Other…", None, None),                     # cancelled
])
def test_mic_open_time_can_be_picked_or_typed(monkeypatch, picked, typed, saved):
    errors = []
    def dialog(*args):
        if "--entry" in args:
            return typed
        if "--error" in args:
            errors.append(args)
        return None
    monkeypatch.setattr(desktop, "zenity", dialog)
    monkeypatch.setattr(desktop, "pick", lambda title, options, current: picked)
    changes = menu.edit_mic_open(store.settings())
    assert changes == (None if saved is None else {"mic_open_minutes": saved})
    assert bool(errors) is (typed is not None and typed.strip() in ("soon", "99999"))


def test_a_typed_mic_open_time_is_shown_ticked_next_time(monkeypatch):
    store.save_settings({"mic_open_minutes": 45})
    shown = {}
    monkeypatch.setattr(desktop, "pick", lambda title, options, current: shown.update(o=options, c=current))
    menu.edit_mic_open(store.settings())
    assert shown["c"] == "open for 45 min after a take" and shown["c"] in shown["o"]


@pytest.mark.parametrize("model, language, answer, saved", [
    ("distil-large-v3.5", "fr", None, None),   # English-only model, French: asked, said No
    ("distil-large-v3.5", "fr", "", "fr"),     # asked, said Yes
    ("distil-large-v3.5", "en", None, "en"),   # English: not asked
    ("distil-large-v3.5", "", None, ""),       # auto-detect: not asked
    ("large-v3-turbo", "fr", None, "fr"),      # a multilingual model: not asked
])
def test_setting_a_language_an_english_only_model_can_t_hear_asks_first(monkeypatch, model, language, answer, saved):
    store.save_settings({"asr_model": model})
    asked = []
    monkeypatch.setattr(desktop, "zenity", lambda *args: asked.append(args) or answer)
    names = {"": "auto-detect", "en": "English (en)", "fr": "French (fr)"}
    monkeypatch.setattr(desktop, "pick", lambda title, options, current: names[language])
    assert menu.edit_language(store.settings()) == (None if saved is None else {"language": saved})
    assert bool(asked) is (model.startswith("distil") and language == "fr")
    assert all("--question" in a for a in asked)


def test_picking_an_english_only_model_with_another_language_set_asks_first(monkeypatch):
    store.save_settings({"language": "fr"})
    monkeypatch.setattr(menu.engine, "model_downloaded", lambda name: True)
    dialogs = []
    def zenity(*args):
        dialogs.append(args)
        return "distil-large-v3.5 (1.5 GB, downloaded): English only: fast and accurate" if "--list" in args else None
    monkeypatch.setattr(desktop, "zenity", zenity)
    monkeypatch.setattr(menu, "switch_asr_model", lambda name: switched.append(name) or False)
    switched = []
    menu.pick_asr_model("large-v3-turbo")
    assert [a[0] for a in dialogs] == ["--list", "--question"] and switched == []  # said No: not switched


def test_quit_button_stops_the_daemon_after_asking(monkeypatch):
    monkeypatch.setattr(menu.shutil, "which", lambda name: "/usr/bin/zenity")
    monkeypatch.setattr(menu, "keys_summary", lambda: "")
    choices = iter(["quit", "quit", None])
    monkeypatch.setattr(menu, "menu_choice", lambda rows: next(choices))
    answers = iter([None, ""])  # first No, then Yes
    monkeypatch.setattr(desktop, "zenity", lambda *args: next(answers))
    ran = []
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(returncode=0))
    menu.settings_menu()
    assert ran == [["systemctl", "--user", "stop", "tuxwhisper"]]  # said No once, then stopped
    assert next(choices) is None  # the menu closed with the daemon: the last choice wasn't shown
    assert notified[-1] == "TuxWhisper stopped"


def test_quit_that_fails_to_stop_the_service_says_so_and_keeps_the_menu(monkeypatch):
    monkeypatch.setattr(menu.shutil, "which", lambda name: "/usr/bin/zenity")
    monkeypatch.setattr(menu, "keys_summary", lambda: "")
    choices = iter(["quit", None])
    monkeypatch.setattr(menu, "menu_choice", lambda rows: next(choices))
    monkeypatch.setattr(desktop, "zenity", lambda *args: "")  # Yes
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(returncode=5))
    menu.settings_menu()
    assert notified[-1] == "Couldn't stop TuxWhisper" and next(choices, "closed") == "closed"  # the menu went on


def test_zenity_menu_quit_button_and_choices(monkeypatch):
    monkeypatch.setattr(desktop, "have_yad", lambda: False)
    replies = iter([(1, menu.QUIT + "\n"), (0, "keys\n"), (1, "")])
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=(reply := next(replies))[0], stdout=reply[1], stderr=""))
    rows = [("keys", "Keys", "F5", "The keys")]
    assert [menu.menu_choice(rows) for _ in range(3)] == ["quit", "keys", None]


def test_yad_menu_shows_the_logo_and_parts_the_groups_with_lines(monkeypatch):
    cmds = []
    monkeypatch.setattr(desktop, "have_yad", lambda: True)
    monkeypatch.setattr(menu.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(
        returncode=0, stdout="<b>Keys &amp; more</b>|F5|The keys\n"))
    rows = [("history", "Recent transcripts", "3 (default)", "Your last"), ("language", "Language", "English", "Spoken"),
            ("keys", "Keys & more", "F5", "The keys")]
    assert menu.menu_choice(rows) == "keys"
    cmd = cmds[0]
    assert f"--image={desktop.LOGO}" in cmd and os.path.exists(desktop.LOGO) and "--sep-column=1" in cmd and f"--sep-value={menu.SEPARATOR}" in cmd
    assert f"--button={menu.QUIT}:2" in cmd
    assert cmd[cmd.index("--") + 1:] == [
        "<b>Recent transcripts</b>", '3 <span alpha="55%">(default)</span>', "Your last", menu.SEPARATOR, "", "",
        "<b>Language</b>", "English", "Spoken", menu.SEPARATOR, "", "", "<b>Keys &amp; more</b>", "F5", "The keys"]


def test_switching_the_model_without_systemd_just_saves_it(monkeypatch):
    def no_systemctl(cmd, **k):
        raise FileNotFoundError("systemctl")
    monkeypatch.setattr(menu.subprocess, "run", no_systemctl)
    assert menu.switch_asr_model("small") is False
    assert store.read_config()["asr_model"] == "small"


def test_recent_transcripts_copy_failure_is_reported(monkeypatch):
    real_add_history("Hello")
    monkeypatch.setattr(desktop, "zenity", lambda *args: "0|")
    monkeypatch.setattr(desktop, "clipboard_set", lambda text, *a: False)
    menu.show_history(store.settings())
    assert notified[-1] == "Couldn't copy it"
