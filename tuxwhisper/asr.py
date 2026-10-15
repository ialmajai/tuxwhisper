"""Push-to-toggle local dictation for Linux (X11 and Wayland).

  tuxwhisper setup      install the system packages, background service and GNOME shortcuts
  tuxwhisper serve      run the daemon (loads the model once and keeps it in memory)
  tuxwhisper toggle     start recording / stop, transcribe and paste at the cursor
  tuxwhisper rewrite    same, but clean up the transcript with a local LLM (Ollama) first
  tuxwhisper save       save the last recording + its transcript for fine-tuning
  tuxwhisper settings   open the settings menu (needs zenity)
  tuxwhisper correct    fix the last transcript and learn from the changes
  tuxwhisper bind       bind the default keys (F5, F3, F4, F6, Alt+F9) as GNOME custom shortcuts
  tuxwhisper uninstall  remove the service, shortcuts and launcher that setup added

From a git clone, `./asr` runs the same commands.
"""
import os
import shutil
import subprocess
import sys

from tuxwhisper import daemon, desktop, menu, store


def migrate_old_files():
    """Move lists and recordings from a pre-0.2 git clone to the standard folders, once."""
    moves = [(os.path.join(store.OLD_REPO, name), os.path.join(store.CONFIG_DIR, name))
             for name in ("vocab.txt", "replacements.txt", "modes.txt")]
    moves.append((os.path.join(store.OLD_REPO, "recordings"), store.RECORDINGS))
    for old, new in moves:
        if os.path.exists(old) and not os.path.exists(new):
            os.makedirs(os.path.dirname(new), exist_ok=True)
            shutil.move(old, new)
            print(f"moved {old} to {new}", flush=True)


def add_library_path():
    """pip installs the CUDA and whisper.cpp libraries inside site-packages, where the dynamic
    loader doesn't look. It reads LD_LIBRARY_PATH only at startup, so restart with it set."""
    import sysconfig
    site = sysconfig.get_paths()["purelib"]
    wanted = [d for d in (f"{site}/nvidia/cudnn/lib", f"{site}/nvidia/cublas/lib") if os.path.isdir(d)]
    if os.path.exists(f"{site}/libwhisper.so.1"):
        wanted.append(site)
    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing = [d for d in wanted if d not in current]
    if missing and not os.environ.get("TUXWHISPER_LIBS_SET"):
        env = dict(os.environ, TUXWHISPER_LIBS_SET="1", LD_LIBRARY_PATH=":".join(missing + current))
        os.execve(sys.executable, [sys.executable, "-m", store.APP, *sys.argv[1:]], env)


def run_script(name, args):
    """Run install.sh or uninstall.sh: shipped inside the package, or at the top of a git clone."""
    packaged = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)
    script = packaged if os.path.exists(packaged) else os.path.join(store.OLD_REPO, name)
    env = dict(os.environ, TUXWHISPER_CMD=store.command_path(), TUXWHISPER_PY=sys.executable)
    if os.path.exists(packaged):
        from tuxwhisper import __version__
        env.update(TUXWHISPER_PACKAGED="1", TUXWHISPER_VERSION=__version__)
    sys.exit(subprocess.call(["bash", script, *args], env=env))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("serve", "settings"):
        migrate_old_files()
    if cmd == "serve":
        add_library_path()
        daemon.serve()
    elif cmd in ("toggle", "rewrite", "save"):
        daemon.send(cmd)
    elif cmd == "settings":
        menu.with_service_env()
        menu.settings_menu()
    elif cmd == "bind":
        desktop.bind_gnome()
    elif cmd == "correct":
        menu.correct_last()
    elif cmd == "download" and len(sys.argv) > 2:
        menu.download_asr_model(sys.argv[2])
    elif cmd == "setup":
        run_script("install.sh", sys.argv[2:])
    elif cmd == "uninstall":
        run_script("uninstall.sh", sys.argv[2:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
