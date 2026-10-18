"""The key commands' side of the daemon's socket: kept apart from the daemon so a key press
imports only this, not the speech or LLM code (tens of ms on every press)."""
import os
import socket
import subprocess
import sys

from tuxwhisper import desktop, store


def sock_path():
    # XDG_RUNTIME_DIR is private to the user (0700); a shared fallback like /tmp would let
    # other local users reach the socket, so refuse to run without it.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:  # an OSError, so callers such as request() handle it instead of exiting
        desktop.notify("TuxWhisper needs a desktop session", body="XDG_RUNTIME_DIR is not set.", level="error")
        raise OSError("XDG_RUNTIME_DIR is not set; run this from a desktop session")
    return os.path.join(runtime, f"{store.APP}.sock")


def request(cmd, timeout=60):
    """Send cmd to the daemon and wait for its reply; None if it isn't running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(sock_path())
            s.sendall(cmd.encode())
            s.shutdown(socket.SHUT_WR)
            return s.recv(16).decode()
    except OSError:
        return None


def send(cmd):
    """Send a key's command to the daemon; it answers by closing the connection."""
    if request(cmd, timeout=5) is None:
        state = subprocess.run(["systemctl", "--user", "is-active", store.APP],
                               capture_output=True, text=True).stdout.strip()
        if state in ("active", "activating"):  # loading its model: the socket isn't up yet
            desktop.notify("TuxWhisper is still starting", body="Try again in a few seconds.", level="warn")
            sys.exit(1)
        # Not running, e.g. systemd gave up after repeated start failures: start it again
        # rather than leave the key doing nothing.
        subprocess.run(["systemctl", "--user", "reset-failed", store.APP], capture_output=True)
        started = subprocess.run(["systemctl", "--user", "start", "--no-block", store.APP],
                                 capture_output=True).returncode == 0
        desktop.notify("Starting TuxWhisper…" if started else f"{store.APP} isn't running",
                       body="Try again in about 15 seconds." if started
                       else "Start it with: systemctl --user start tuxwhisper",
                       level="info" if started else "error")
        sys.exit(1)
