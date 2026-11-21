"""Tests for the key commands' side of the socket."""
import subprocess
import sys
import types

import pytest
from conftest import notified

from tuxwhisper import client


def fake_systemctl(monkeypatch, state):
    """The daemon isn't answering and systemd reports it as state; the commands run are returned."""
    monkeypatch.setattr(client, "request", lambda cmd, timeout=5: None)
    ran = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: ran.append(cmd) or types.SimpleNamespace(
        returncode=0, stdout=state + "\n"))
    with pytest.raises(SystemExit):
        client.send("toggle")
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


def test_cli_key_commands_import_only_the_socket_client():
    code = ("import sys; from tuxwhisper import asr, client; "
            "print([m for m in ('tuxwhisper.engine', 'tuxwhisper.llm', 'tuxwhisper.menu', "
            "'tuxwhisper.daemon', 'tuxwhisper.settings', 'tuxwhisper.desktop', 'dataclasses', 'subprocess', "
            "'urllib.request', 'http.client', 'tomllib', 'json', 'csv', 're', 'shutil') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


def test_command_delivered_but_unanswered_in_time_is_not_a_daemon_that_is_down(tmp_path):
    """The daemon took the word (e.g. a long retype): the key must not report it as not running
    or start it again, as the Rust client doesn't."""
    import socket
    import threading
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(client.sock_path())
    srv.listen()
    held = []
    threading.Thread(target=lambda: held.append(srv.accept()[0]), daemon=True).start()  # never answers
    try:
        assert client.request("retype", timeout=0.1) == ""
    finally:
        srv.close()
        for c in held:
            c.close()
