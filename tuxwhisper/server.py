"""The daemon's socket: the key commands reach the running dictation through it."""
import fcntl
import os
import socket
import sys
import threading
import traceback

from tuxwhisper import desktop, store
from tuxwhisper.client import sock_path
from tuxwhisper.daemon import Dictation


def daemon_running(sock: str) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(sock)
        return True
    except OSError:
        return False


def serve() -> None:
    sock = sock_path()
    # Held for the daemon's lifetime: a second daemon started during the first one's model
    # load (~15 s, before the socket exists) would otherwise steal the socket and orphan it.
    lock = open(f"{sock}.lock", "a")  # noqa: SIM115 - deliberately kept open
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"{store.APP} daemon is already running")
    if daemon_running(sock):  # don't steal the socket from a live daemon
        sys.exit(f"{store.APP} daemon is already running")
    desktop.close_stale_notifications()  # left open by a daemon that crashed or restarted mid-recording
    d = Dictation.create()
    if os.path.exists(sock):  # stale socket from a daemon that didn't exit cleanly
        os.unlink(sock)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock)
    os.chmod(sock, 0o600)
    srv.listen()
    while True:
        conn, _ = srv.accept()
        # Each in its own thread: a client that sends nothing, or a long retype, must not
        # hold up the next key press.
        threading.Thread(target=handle, args=(d, conn), daemon=True).start()


def handle(d: Dictation, conn: socket.socket) -> None:
    """Run one connection's command."""
    try:
        with conn:
            conn.settimeout(2)
            cmd = conn.recv(64).decode(errors="replace").strip()
            if cmd == "retype":  # the caller waits, so its next dialog can't steal the focus
                conn.settimeout(None)
                conn.sendall(b"ok" if d.retype() else b"no")
                return
        if cmd == "toggle":
            d.press()
        elif cmd == "rewrite":
            d.press(rewrite=True)
        elif cmd == "save":
            d.save()
    except Exception:  # noqa: BLE001 - one failed command must not stop dictation
        traceback.print_exc()
        desktop.notify("TuxWhisper command failed", body="See the log: journalctl --user -u tuxwhisper",
                       level="error")
