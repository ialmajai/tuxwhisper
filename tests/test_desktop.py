"""Tests for notifications, dialogs, the clipboard and GNOME shortcuts."""
import os
import subprocess
import time
import types

import pytest
from conftest import real_close_notification, real_notify

from tuxwhisper import desktop


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


@pytest.mark.parametrize("offered, saved", [
    (["TARGETS", "image/png", "text/html"], ("image/png", b"data of image/png")),  # a screenshot
    (["TARGETS", "UTF8_STRING", "text/html"], ("text", b"the text")),    # rich text: as text
    (["TARGETS", "x-special/gnome-copied-files"], ("x-special/gnome-copied-files",
                                                   b"data of x-special/gnome-copied-files")),
    ([], None),                                                                    # empty
])
def test_clipboard_save_keeps_images_and_other_formats_not_only_text(monkeypatch, offered, saved):
    monkeypatch.setattr(desktop, "USE_XCLIP", True)
    def run(cmd, **k):
        kind = cmd[cmd.index("-t") + 1]
        out = "\n".join(offered).encode() if kind == "TARGETS" else f"data of {kind}".encode()
        return types.SimpleNamespace(returncode=0 if offered else 1, stdout=out)
    monkeypatch.setattr(desktop.subprocess, "run", run)
    monkeypatch.setattr(desktop, "clipboard_get", lambda sel="clipboard": "the text")
    assert desktop.clipboard_save() == saved


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


@pytest.mark.parametrize("out, window", [("uint32 500\n", 0.7), ("uint32 900\n", 1.1), ("", 0.7)])
def test_repeat_window_follows_the_desktop_repeat_delay(monkeypatch, out, window):
    monkeypatch.setattr(desktop.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert desktop.repeat_window(0.7) == pytest.approx(window)


@pytest.mark.parametrize("level, body, timeout", [
    ("info", None, "1500"), ("info", "b", "6000"), ("warn", None, "8000"), ("error", "b", "8000")])
def test_notify_timeout_follows_its_level(monkeypatch, level, body, timeout):
    cmds = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop, "spawn", cmds.append)
    real_notify("Title", body=body, level=level)
    assert cmds[0][cmds[0].index("-t") + 1] == timeout


def test_spawned_helper_is_reaped_without_waiting_for_it():
    p = []
    real = desktop.subprocess.Popen
    with pytest.MonkeyPatch.context() as m:
        m.setattr(desktop.subprocess, "Popen", lambda cmd, **k: p.append(real(cmd, **k)) or p[-1])
        desktop.spawn(["sleep", "0.2"])
    assert p[0].returncode is None  # spawn returned at once
    deadline = time.monotonic() + 5
    while p[0].returncode is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert p[0].returncode == 0  # waited for: no zombie left behind


def test_persistent_notify_stays_and_can_replace(monkeypatch, tmp_path):
    cmds = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: cmds.append(cmd) or types.SimpleNamespace(
        stdout="9\n"))
    assert real_notify("Transcribing…", persistent=True, icon="mic", replace="5") == "9"
    cmd = cmds[0]
    assert cmd[cmd.index("-t") + 1] == "0" and cmd[cmd.index("-u") + 1] == "normal"
    assert cmd[cmd.index("-r") + 1] == "5" and cmd[cmd.index("-i") + 1] == "mic"


def test_notification_is_not_changed_while_its_banner_slides_in(monkeypatch):
    shown_at = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop.subprocess, "run", lambda cmd, **k: shown_at.append(time.monotonic()) or
                        types.SimpleNamespace(stdout="11\n"))
    real_notify("Recording…", persistent=True)
    real_notify("Transcribing…", persistent=True, replace="11")
    assert shown_at[1] - shown_at[0] >= desktop.BANNER_SETTLE - 0.05


def test_notification_closed_while_its_banner_slides_in_closes_once_settled(monkeypatch):
    closed_at = []
    monkeypatch.setattr(desktop.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(desktop, "spawn", lambda cmd: closed_at.append((cmd[-1], time.monotonic())))
    monkeypatch.setitem(desktop._shown, "12", time.monotonic())
    start = time.monotonic()
    real_close_notification("12")
    assert closed_at == [] and time.monotonic() - start < 0.1  # the caller isn't held up
    deadline = start + 5
    while not closed_at and time.monotonic() < deadline:
        time.sleep(0.02)
    assert closed_at[0][0] == "12" and closed_at[0][1] - start >= desktop.BANNER_SETTLE - 0.05
    real_close_notification("13")  # long settled (or never tracked): closes at once
    assert closed_at[-1][0] == "13"


def test_shortcut_installed_under_a_path_with_a_space_is_recognised(monkeypatch):
    printed = {"custom-keybindings": "['/k/tuxwhisper-toggle/']", "binding": "'F5'",
               "command": "\"'/home/a b/asr' toggle\""}  # gsettings double-quotes a value with a '
    monkeypatch.setattr(desktop.subprocess, "run", lambda args, **k: types.SimpleNamespace(
        returncode=0, stdout=printed[args[-1]] + "\n", stderr=""))
    shortcuts = desktop.gnome_shortcuts()
    assert shortcuts == {"/k/tuxwhisper-toggle/": ("F5", "'/home/a b/asr' toggle")}
    assert desktop.our_shortcut(shortcuts, "toggle") == ("/k/tuxwhisper-toggle/", "F5")


def test_plural():
    assert desktop.plural(1, "word", "words") == "1 word"
    assert desktop.plural(0, "word", "words") == "0 words"
    assert desktop.plural(3, "recording", "recordings") == "3 recordings"


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
