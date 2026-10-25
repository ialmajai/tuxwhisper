"""Tests for the Rust key-press client (client-rs), run against a fake daemon socket."""
import os
import socket
import subprocess
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = os.path.join(ROOT, "client-rs", "target", "release", "tuxwhisper-key")
pytestmark = pytest.mark.skipif(not os.access(KEY, os.X_OK), reason="client-rs isn't built (cargo build --release)")


def fake_daemon(runtime, reply_after=0.0):
    """Listen on the daemon's socket in runtime; returns the list each received command goes to."""
    got = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(os.path.join(runtime, "tuxwhisper.sock"))
    server.listen()
    def serve():
        conn, _ = server.accept()
        with conn:
            got.append(conn.recv(64).decode())
            time.sleep(reply_after)  # the daemon closes the connection to answer
        server.close()
    threading.Thread(target=serve, daemon=True).start()
    return got


def python_client(tmp_path):
    """A stand-in for the Python client: notes how it was run, in tmp_path/fallback."""
    script = tmp_path / "python-client"
    script.write_text('#!/bin/sh\necho "$TUXWHISPER_SLOW $*" > "$(dirname "$0")/fallback"\n')
    script.chmod(0o755)
    return str(script)


def run(args, tmp_path, **env):
    return subprocess.run(args, env={"PATH": os.environ["PATH"], "XDG_RUNTIME_DIR": str(tmp_path),
                                     "TUXWHISPER_CMD": python_client(tmp_path), **env},
                          capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize("cmd", ["toggle", "rewrite", "save"])
def test_key_reaches_the_daemon_without_python(tmp_path, cmd):
    got = fake_daemon(str(tmp_path))
    r = run([KEY, cmd], tmp_path)
    assert r.returncode == 0 and got == [cmd]
    assert not (tmp_path / "fallback").exists()


def test_no_daemon_hands_the_key_to_the_python_client(tmp_path):
    r = run([KEY, "toggle"], tmp_path)  # no socket: the Python client starts the daemon or says why
    assert r.returncode == 0
    assert (tmp_path / "fallback").read_text().strip() == "1 toggle"


def test_slow_reply_is_not_followed_by_a_second_send(tmp_path):
    got = fake_daemon(str(tmp_path), reply_after=0.5)
    r = run([KEY, "toggle"], tmp_path)
    assert r.returncode == 0 and got == ["toggle"]  # a second toggle would undo the first
    assert not (tmp_path / "fallback").exists()


def test_other_commands_go_to_the_python_client(tmp_path):
    got = fake_daemon(str(tmp_path))
    run([KEY, "settings"], tmp_path)
    assert got == [] and (tmp_path / "fallback").read_text().strip() == "1 settings"


def test_wrapper_sends_key_presses_through_the_rust_client(tmp_path):
    got = fake_daemon(str(tmp_path))
    r = run([os.path.join(ROOT, "asr"), "toggle"], tmp_path, TUXWHISPER_CMD="")
    assert r.returncode == 0 and got == ["toggle"]
