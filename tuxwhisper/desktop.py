"""Desktop integration: notifications, sounds, the clipboard, GNOME shortcuts and dialogs."""
import ast
import contextlib
import functools
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from tuxwhisper import store

# How long a notice stays up, in ms: a bare title is read at a glance, a body takes longer,
# and a warning or error should stay long enough to read even when looking away.
NOTIFY_TIMEOUTS = {"info": 1500, "info+body": 6000, "warn": 8000, "error": 8000}
NOTIFY_ICONS = {"warn": "dialog-warning", "error": "dialog-error"}
# GNOME Shell (49) breaks if a notification is replaced or closed while its banner slides in:
# it then shows no banners from any app until the next login. So a persistent notification is
# left alone for this long (seconds) after it was shown or changed.
BANNER_SETTLE = 0.5
_shown: dict[str, float] = {}  # persistent notification ID -> when it was last shown or changed


def _unsettled(nid: str | None) -> float:
    """Seconds until notification nid may be changed or closed."""
    return max(0.0, _shown.get(nid, 0.0) + BANNER_SETTLE - time.monotonic()) if nid else 0.0


def spawn(cmd: list[str]) -> None:
    """Run a helper without waiting for it, but reap it when it exits: a finished child left
    unwaited stays in the service as a zombie until the next subprocess call."""
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    threading.Thread(target=p.wait, daemon=True).start()


def notify(msg: str, persistent: bool = False, body: str | None = None, level: str = "info",
           icon: str | None = None, replace: str | None = None) -> str | None:
    """Show a desktop notification; level is "info", "warn" or "error". persistent=True keeps
    it on screen until close_notification() is called with the returned ID; replace=<ID>
    swaps the text of that one in place, so a status changes without a new popup."""
    extra = [body] if body else []
    icon = icon or NOTIFY_ICONS.get(level)
    opts = ["-a", store.APP, "-h", "boolean:transient:true", *(["-i", icon] if icon else [])]
    if not shutil.which("notify-send"):  # notifications are optional
        print(msg, flush=True)
        return None
    if persistent:
        time.sleep(_unsettled(replace))
        try:
            r = subprocess.run(["notify-send", *opts, "-u", "normal", "-t", "0", "-p",
                                *(["-r", replace] if replace else []), msg, *extra],
                               capture_output=True, text=True, timeout=2)
            nid = r.stdout.strip() or None
            if nid:
                _shown[nid] = time.monotonic()
            track_notification(nid, open_=True)
            return nid
        except subprocess.TimeoutExpired:
            return None
    timeout = NOTIFY_TIMEOUTS["info+body" if level == "info" and body else level]
    spawn(["notify-send", *opts, "-t", str(timeout), msg, *extra])
    return None


def plural(n: int, one: str, many: str) -> str:
    """"1 recording", "3 recordings"."""
    return f"{n} {one if n == 1 else many}"


def settings_shortcut() -> str | None:
    """The GNOME shortcut bound to the settings command, e.g. "Alt+F9", or None."""
    ours = our_shortcut(gnome_shortcuts() or {}, "settings")
    return key_label(ours[1]) if ours else None


def repeat_window(default: float) -> float:
    """How soon a second press counts as the key's auto-repeat: the first repeat comes after
    the desktop's repeat delay (GNOME's default 500 ms), so allow that plus a margin. Users
    with a longer delay (common for accessibility) would otherwise have a hold cut short.
    default when the desktop doesn't say."""
    try:
        r = subprocess.run(["gsettings", "get", "org.gnome.desktop.peripherals.keyboard", "delay"],
                           capture_output=True, text=True, timeout=2)
        return int(r.stdout.split()[-1]) / 1000 + 0.2
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return default


GNOME_KEYS = {"toggle": "F5", "rewrite": "F3", "save": "F4", "correct": "F6", "settings": "<Alt>F9"}
KEY_ACTIONS = {"toggle": "Dictate", "rewrite": "Dictate with LLM cleanup",
               "save": "Save the last recording", "settings": "Open this settings menu",
               "correct": "Correct the last transcript"}
KEY_SUGGESTIONS = ["F5", "F3", "F4", "F6", "F7", "F8", "F9", "F10", "F12", "Alt+F9",
                   "Ctrl+Alt+D", "Super+H"]
MEDIA_KEYS = "org.gnome.settings-daemon.plugins.media-keys"


def key_label(binding: str) -> str:
    """GNOME's "<Control><Alt>d" as "Ctrl+Alt+D"."""
    mods, key = re.match(r"((?:<\w+>)*)(.*)", binding).groups()
    mods = re.sub(r"<(\w+)>", r"\1+", mods).replace("Primary+", "Ctrl+").replace("Control+", "Ctrl+")
    return mods + key[:1].upper() + key[1:]


def parse_key(text: str) -> str | None:
    """"Ctrl+Alt+D" or "F6" as GNOME's "<Control><Alt>d" / "F6"; None if it isn't a usable key."""
    *mods, key = [part.strip() for part in text.split("+")]
    names = {"ctrl": "<Control>", "control": "<Control>", "alt": "<Alt>", "shift": "<Shift>",
             "super": "<Super>"}
    special = {"space": "space", "insert": "Insert", "home": "Home", "end": "End", "pause": "Pause",
               "print": "Print", "menu": "Menu"}
    if any(m.lower() not in names for m in mods):
        return None
    if re.fullmatch(r"[Ff]([1-9]|1[0-9]|2[0-4])", key):
        key = key.upper()
    elif re.fullmatch(r"[A-Za-z0-9]", key) or key.lower() in special:
        if not mods:  # a bare letter or space would stop you typing it
            return None
        key = special.get(key.lower(), key.lower())
    else:
        return None
    return "".join(names[m.lower()] for m in mods) + key


def gnome_shortcuts() -> dict[str, tuple[str, str]] | None:
    """Every GNOME custom shortcut as {path: (binding, command)}, or None when not on GNOME."""
    def get(*args):
        r = subprocess.run(["gsettings", "get", *args], capture_output=True, text=True, timeout=5)
        if r.returncode:
            raise FileNotFoundError(r.stderr)
        return r.stdout.strip()
    try:
        paths = re.findall(r"'([^']+)'", get(MEDIA_KEYS, "custom-keybindings"))
        return {p: tuple(gvariant_string(get(f"{MEDIA_KEYS}.custom-keybinding:{p}", k)) for k in ("binding", "command"))
                for p in paths}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def gvariant_string(printed: str) -> str:
    """A string as gsettings prints it: in single quotes, or double ones if it holds a single
    quote (a path such as '/home/a b/asr', quoted for the shell, comes back as "'/home/a b/asr' x")."""
    try:
        value = ast.literal_eval(printed)
    except (ValueError, SyntaxError):
        return printed
    return value if isinstance(value, str) else printed


def runs_command(command: str, cmd: str) -> bool:
    """Whether a shortcut's command line runs TuxWhisper's cmd: `<path>/asr cmd` from a git
    clone or `<path>/tuxwhisper cmd` when installed, paths quoted or not. Exact, so a shortcut
    running some other program whose name merely ends in "asr" isn't mistaken for ours."""
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return len(words) == 2 and os.path.basename(words[0]) in ("asr", store.APP) and words[1] == cmd


def our_shortcut(shortcuts: dict[str, tuple[str, str]], cmd: str) -> tuple[str, str] | None:
    """(path, binding) of the shortcut running cmd, from gnome_shortcuts(), or None."""
    return next(((p, b) for p, (b, c) in shortcuts.items() if runs_command(c, cmd)), None)


def set_shortcut(cmd: str, binding: str, path: str | None, paths: list[str]) -> str:
    """Point TuxWhisper's shortcut for cmd at binding, creating the shortcut if needed; its path."""
    def gs(*args):
        subprocess.run(["gsettings", "set", *args], check=True, timeout=5)
    if path is None:
        path = f"/{MEDIA_KEYS.replace('.', '/')}/custom-keybindings/tuxwhisper-{cmd}/"
        while path in paths:
            path = path[:-1] + "-1/"
        gs(f"{MEDIA_KEYS}.custom-keybinding:{path}", "name", f"TuxWhisper {cmd}")
        gs(f"{MEDIA_KEYS}.custom-keybinding:{path}", "command", f"{shlex.quote(store.command_path())} {cmd}")
        gs(MEDIA_KEYS, "custom-keybindings", str([*paths, path]))
    gs(f"{MEDIA_KEYS}.custom-keybinding:{path}", "binding", binding)
    return path


def bind_gnome() -> None:
    """Add GNOME custom shortcuts for the default keys, keeping any the user already changed."""
    shortcuts = gnome_shortcuts()
    if shortcuts is None:
        sys.exit("GNOME's gsettings not found; bind the keys by hand (see docs/other-desktops.md).")
    paths, cmd_path = list(shortcuts), shlex.quote(store.command_path())
    for cmd, key in GNOME_KEYS.items():
        ours = our_shortcut(shortcuts, cmd)
        if ours:  # keep the key the user chose; only update the path, in case the repo moved
            subprocess.run(["gsettings", "set", f"{MEDIA_KEYS}.custom-keybinding:{ours[0]}", "command",
                            f"{cmd_path} {cmd}"], check=True, timeout=5)
            print(f"{key_label(ours[1]):7} {cmd_path} {cmd} (already bound)")
            continue
        taken = next((c for b, c in shortcuts.values() if b == key), None)
        if taken:
            print(f"{key_label(key):7} skipped: already runs {taken}. Bind {cmd_path} {cmd} to another key.")
            continue
        paths.append(set_shortcut(cmd, key, None, paths))
        print(f"{key_label(key):7} {cmd_path} {cmd}")


def sound(name: str) -> None:
    """Play a freedesktop theme sound; silently skipped if no player or file is found."""
    path = f"/usr/share/sounds/freedesktop/stereo/{name}.oga"
    volume = store.settings().sound_volume / 100
    if shutil.which("pw-play"):
        cmd = ["pw-play", f"--volume={volume}", path]
    elif shutil.which("paplay"):
        cmd = ["paplay", f"--volume={int(volume * 65536)}", path]
    else:
        return
    if store.settings().sounds and os.path.exists(path):
        spawn(cmd)


def open_notifications_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(runtime, f"{store.APP}-notifications") if runtime else None


_notifications_lock = threading.Lock()


def track_notification(nid, open_):
    """Remember which persistent notifications are on screen, so a daemon that died while one
    was open can close it on its next start (GNOME keeps them, even on the lock screen)."""
    path = open_notifications_path()
    if not nid or not path:
        return
    with _notifications_lock:
        try:
            with open(path) as f:
                ids = set(f.read().split())
        except FileNotFoundError:
            ids = set()
        ids = ids | {nid} if open_ else ids - {nid}
        with open(path, "w") as f:
            f.write("\n".join(sorted(ids)))


def close_stale_notifications() -> None:
    path = open_notifications_path()
    try:
        with open(path) as f:
            for nid in f.read().split():
                close_notification(nid)
    except (FileNotFoundError, TypeError):
        pass


def close_notification(nid: str | None) -> None:
    """Close a persistent notification; one shown under BANNER_SETTLE ago closes once it has
    settled, without holding up the caller."""
    track_notification(nid, open_=False)
    if nid and shutil.which("gdbus"):
        wait = _unsettled(nid)
        _shown.pop(nid, None)
        cmd = ["gdbus", "call", "--session", "--dest", "org.freedesktop.Notifications",
               "--object-path", "/org/freedesktop/Notifications",
               "--method", "org.freedesktop.Notifications.CloseNotification", nid]
        if wait:
            threading.Timer(wait, spawn, [cmd]).start()
        else:
            spawn(cmd)


# xclip works on X11 and on Wayland desktops with XWayland (GNOME, KDE, most others);
# wl-clipboard covers Wayland sessions without XWayland.
USE_XCLIP = bool(os.environ.get("DISPLAY") and shutil.which("xclip"))


def clipboard_get(selection: str = "clipboard") -> str | None:
    """Text in the clipboard or primary selection, or None if empty or not text (e.g. an image)."""
    if USE_XCLIP:
        cmd = ["xclip", "-selection", selection, "-o", "-t", "UTF8_STRING"]
    else:
        cmd = ["wl-paste", "--no-newline", "--type", "text"] + (["--primary"] if selection == "primary" else [])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return r.stdout.decode(errors="replace") if r.returncode == 0 else None


def clipboard_types(selection: str = "clipboard") -> list[str]:
    """The formats the clipboard or primary selection offers; [] if it's empty."""
    if USE_XCLIP:
        cmd = ["xclip", "-selection", selection, "-o", "-t", "TARGETS"]
    else:
        cmd = ["wl-paste", "--list-types"] + (["--primary"] if selection == "primary" else [])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return r.stdout.decode(errors="replace").split() if r.returncode == 0 else []


def clipboard_save(selection: str = "clipboard") -> tuple[str, bytes] | None:
    """The selection's contents as (format, data), for clipboard_restore(): an image as an
    image, else text as text, else its first format's data; None if empty or unreadable.
    One format is kept, so rich text comes back as plain text."""
    types = clipboard_types(selection)
    kind = next((t for t in types if t.startswith("image/")), None)
    if kind is None and any(t == "UTF8_STRING" or t.startswith("text/plain") for t in types):
        text = clipboard_get(selection)
        return ("text", text.encode()) if text is not None else None
    kind = kind or next((t for t in types if "/" in t), None)  # a MIME type, not an X11 atom
    if kind is None:
        return None
    if USE_XCLIP:
        cmd = ["xclip", "-selection", selection, "-o", "-t", kind]
    else:
        cmd = ["wl-paste", "--type", kind] + (["--primary"] if selection == "primary" else [])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=2)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return (kind, r.stdout) if r.returncode == 0 and r.stdout else None


def clipboard_restore(saved: tuple[str, bytes], selection: str = "clipboard") -> bool:
    """Put back what clipboard_save() read."""
    kind, data = saved
    if kind == "text":
        return clipboard_set(data.decode(), selection)
    if USE_XCLIP:
        cmd = ["xclip", "-selection", selection, "-t", kind]
    else:
        cmd = ["wl-copy", "--type", kind] + (["--primary"] if selection == "primary" else [])
    try:
        subprocess.run(cmd, input=data, check=True, timeout=2)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"clipboard: {e}", flush=True)
        return False


def clipboard_set(text: str, selection: str = "clipboard") -> bool:
    if USE_XCLIP:
        cmd = ["xclip", "-selection", selection]
    else:
        cmd = ["wl-copy"] + (["--primary"] if selection == "primary" else [])
    try:
        subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"clipboard: {e}", flush=True)
        return False


@functools.cache
def have_yad() -> bool:
    return shutil.which("yad") is not None


# The app name GNOME matches to the launcher, so dialogs show its icon instead of yad's.
YAD_APP = ["--name=tuxwhisper-settings", "--window-icon=tuxwhisper"]
YAD_ICONS = {"--info": "dialog-information", "--error": "dialog-error", "--question": "dialog-question"}


def yad_args(args):
    """zenity's options in yad's spelling: yad has no dialog-type flags or button-label options."""
    out, ok, cancel = [], "OK", "Cancel"
    for a in args:
        opt, _, value = a.partition("=")
        if opt == "--ok-label":
            ok = value
        elif opt == "--cancel-label":
            cancel = value
        elif opt in YAD_ICONS:
            out.append(f"--image={YAD_ICONS[opt]}")
        else:
            out.append(a)
    if "--info" in args or "--error" in args:
        return [*out, f"--button={ok}:0"]
    if "--question" in args:
        return [*out, "--button=No:1", "--button=Yes:0"]
    return [*out, f"--button={cancel}:1", f"--button={ok}:0"]


# Exit codes for Cancel, No or closing the window: 1 in zenity; in yad, 1 for our Cancel and No
# buttons (see yad_args) and 252 for closing the window. Anything else means the dialog failed.
DIALOG_CANCELLED = (1, 252)


def zenity(*args: str) -> str | None:
    """Show a zenity dialog, or the same dialog in yad when it's installed: yad opens about
    a second sooner than zenity 4. None if cancelled; raises RuntimeError if the dialog
    couldn't be shown, so a missing or broken tool isn't mistaken for the user's Cancel."""
    rows = []
    if "--list" in args:  # rows follow the options; "--" keeps a row like "- item" from reading as one
        first_row = next((i for i, a in enumerate(args) if not a.startswith("-")), len(args))
        args, rows = args[:first_row], ["--", *args[first_row:]]
    cmd = ["yad", *YAD_APP, "--center", *yad_args(args)] if have_yad() else ["zenity", *args]
    cmd += rows
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{cmd[0]} isn't installed") from e
    if r.returncode in DIALOG_CANCELLED:
        return None
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}): {r.stderr.strip()}")
    out = r.stdout.strip()
    return out.removesuffix("|") if have_yad() and "--list" in args else out  # yad ends list rows with |


def fit(headers, rows):
    """--width/--height for a list dialog, sized to its longest visible cell in each column;
    fixed sizes cut long values off. zenity (GTK 4) pads columns and the dialog more than yad."""
    rows = list(rows)
    chars = sum(max(len(str(c)) for c in col) for col in zip(headers, *rows))
    pad, row, extra = (20, 29, 170) if have_yad() else (50, 27, 250)
    width = chars * 8 + pad * len(headers) + 60
    height = row * len(rows) + extra
    return f"--width={min(max(width, 360), 1400)}", f"--height={min(max(height, 300), 800)}"


def edit_list(path: str, title: str, help_text: str) -> None:
    """Edit a vocab/replacements file in a dialog; the file keeps its comments."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(help_text)
    text = zenity("--text-info", "--editable", f"--filename={path}", f"--title={title}",
                  "--width=520", "--height=420", "--ok-label=Save")
    if text is not None:
        with open(path, "w") as f:
            f.write(text + "\n")


def edit_text(text: str, title: str, ok_label: str, explain: str | None = None) -> str | None:
    """Show text in an editable box, with explain above it; the edited text, or None if
    cancelled. zenity's text-info has no --text, so there explain goes in the title. The draft lives in
    the private runtime dir, since it's dictated text, and never anywhere else; a file of its
    own, since two dialogs can be open at once."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        raise OSError("XDG_RUNTIME_DIR is not set; run this from a desktop session")
    if explain and not have_yad():
        title = f"{title}: {explain}"
    fd, draft = tempfile.mkstemp(prefix=f"{store.APP}-draft-", suffix=".txt", dir=runtime)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return zenity("--text-info", "--editable", f"--filename={draft}", f"--title={title}",
                      *([f"--text={explain}"] if explain and have_yad() else []),
                      "--width=560", "--height=260", f"--ok-label={ok_label}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(draft)


def pick(title: str, options: list[str], current: str | None) -> str | None:
    # A plain list, not --radiolist: there, clicking a row's text doesn't tick its button,
    # so Select silently returned the old value.
    return zenity("--list", f"--title={title}", *fit(("Current", "Value"), [("✓", o) for o in options]),
                  "--column=Current", "--column=Value", "--print-column=2",
                  *[x for o in options for x in ("✓" if o == current else "", o)])
