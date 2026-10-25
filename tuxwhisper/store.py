"""Paths, settings and the files shared by the daemon, the menu and the CLI."""
import contextlib
import csv
import fcntl
import io
import json
import os
import re
import shutil
import time
import tomllib

APP = "tuxwhisper"
PASTE_KEYS = {"shift+insert": ["KEY_LEFTSHIFT", "KEY_INSERT"],
              "ctrl+v": ["KEY_LEFTCTRL", "KEY_V"],
              "ctrl+shift+v": ["KEY_LEFTCTRL", "KEY_LEFTSHIFT", "KEY_V"]}
CONFIG_DIR = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), APP)
DATA_DIR = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), APP)
# The settings menu writes here; its values override the environment variables above.
CONFIG = os.path.join(CONFIG_DIR, "config.toml")
RATE = 16000
RECORDINGS = os.path.join(DATA_DIR, "recordings")
# Names and jargon Whisper should spell right, one per line; re-read on every take.
VOCAB_HEADER = "# Names and jargon Whisper should spell right, one per line.\n"
REPLACEMENTS_HEADER = "# One rule per line: spoken words => written text. \\n is a line break.\n"
VOCAB = os.path.join(CONFIG_DIR, "vocab.txt")
# "spoken words => text" rules applied before pasting; re-read on every take.
REPLACEMENTS = os.path.join(CONFIG_DIR, "replacements.txt")
# Before 0.2 these lived in the git clone, next to asr.py.
OLD_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@contextlib.contextmanager
def file_lock(path):
    """Serialise a read-change-write of path between the daemon and the menu or CLI, which
    otherwise each write their own stale copy and lose the other's change."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        yield
        return
    with open(os.path.join(runtime, f"{APP}-{os.path.basename(path)}.lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


_instance_locks = []  # held open (and so locked) until the process exits


def single_instance(name: str) -> bool:
    """Whether this is the only process running name (e.g. a dialog): False if another one
    holds its lock. A key held down re-runs its shortcut, which would open a window per repeat."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        return True
    f = open(os.path.join(runtime, f"{APP}-{name}.lock"), "a")  # noqa: SIM115 - held open: it is the lock
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return False
    _instance_locks.append(f)
    return True


def atomic_write(path: str, text: str) -> None:
    """Replace path's contents in one step: a reader never sees a half-written file, and a
    crash mid-write leaves the old file instead of a truncated one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_config() -> dict[str, object]:
    """Only the settings changed in the menu. An unreadable file counts as empty (defaults),
    but is reported once, and save_settings keeps a copy of it rather than overwriting it."""
    try:
        with open(CONFIG, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        if not getattr(read_config, "warned", False):
            read_config.warned = True
            print(f"{CONFIG} is unreadable ({e}); using default settings", flush=True)
            # Imported here, not at the top: desktop imports store.
            from tuxwhisper import desktop
            desktop.notify("Settings file is unreadable; using defaults", body=f"Check {CONFIG}.", level="warn")
        return {}


def settings():  # -> Settings (see tuxwhisper.settings)
    """Live settings: environment defaults overridden by the config file, re-read on every use."""
    from tuxwhisper.settings import Settings
    return Settings.from_config(read_config())


def __getattr__(name):
    """store.Settings, imported on first use: dataclasses costs a key press ~6 ms."""
    if name == "Settings":
        from tuxwhisper.settings import Settings
        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


MODES_FILE = os.path.join(CONFIG_DIR, "modes.txt")


def vocab() -> str | None:
    try:
        with open(VOCAB) as f:
            words = [w.strip() for w in f if w.strip() and not w.startswith("#")]
    except FileNotFoundError:
        return None
    return ", ".join(words) or None


def apply_replacements(text: str) -> str:
    try:
        with open(REPLACEMENTS) as f:
            rules = [line.split("=>", 1) for line in f if "=>" in line and not line.startswith("#")]
    except FileNotFoundError:
        return text
    for spoken, written in rules:
        spoken, written = spoken.strip(), written.strip().replace("\\n", "\n")
        # Whole words, any case. A line-break rule also eats the punctuation and spaces
        # Whisper puts around it ("Hello. New line. World." -> "Hello.\nWorld.").
        pattern = r"\b" + re.escape(spoken) + r"\b"
        if not written.strip():
            pattern = r"[ ,]*" + pattern + r"[.,]?[ ]*"
        text = re.sub(pattern, lambda _, w=written: w, text, flags=re.IGNORECASE)
    return text


def dataset_size(meta: str) -> tuple[int, float]:
    """Number of saved takes and their total length in seconds."""
    takes, frames = dataset_frames(meta)
    return takes, frames / RATE


WAV_HEADER = 44  # bytes before the samples in the files write_recording makes; 2 bytes a sample


def dataset_frames(meta: str) -> tuple[int, int]:
    """(takes, samples) of the recordings listed in meta, from the files' sizes (no need to
    open each one)."""
    takes, frames = 0, 0
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            try:
                frames += max(0, os.path.getsize(os.path.join(RECORDINGS, row["file_name"])) - WAV_HEADER) // 2
                takes += 1
            except (OSError, KeyError, TypeError):  # file deleted by hand
                pass
    return takes, frames


def history_path() -> str | None:
    # The runtime dir is private to the user and emptied at logout, so dictated text
    # doesn't pile up on disk.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(runtime, f"{APP}-history.json") if runtime else None


def load_history() -> list[dict]:
    try:
        with open(history_path()) as f:
            return json.load(f)
    except (TypeError, OSError, ValueError):
        return []


def add_history(text: str, rewritten: bool = False, typed: bool = False) -> dict | None:
    """Keep the last 20 transcripts for "Recent transcripts" and "Correct last transcript".
    typed: it was typed at the cursor (not just copied), so a correction may erase and retype it."""
    if history_path():
        entry = {"time": time.strftime("%H:%M"), "text": text, "rewritten": rewritten,
                 "typed": typed, "at": time.time()}
        with file_lock(history_path()):
            save_history((load_history() + [entry])[-20:])
        return entry
    return None


def save_history(items: list[dict]) -> None:
    atomic_write(history_path(), json.dumps(items))


def update_history_entry(entry: dict, **changes: object) -> None:
    """Change one history entry, found again by its time and text: other takes may have been
    added since it was read."""
    with file_lock(history_path()):
        items = load_history()
        match = next((i for i in items if i.get("at") == entry.get("at") and i["text"] == entry["text"]), None)
        if match is not None:
            match.update(changes)
            save_history(items)


def mark_last_saved(name: str) -> None:
    """Note that F4 saved the last take as recordings/name, so a correction can fix its transcript."""
    items = load_history()
    if items:
        update_history_entry(items[-1], recording=name)


def retype_path() -> str:
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR", ""), f"{APP}-retype.json")


# Every language Whisper knows, roughly by total speakers (native and second language).
LANGUAGES = {
    "en": "English", "zh": "Chinese", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "ar": "Arabic", "bn": "Bengali", "pt": "Portuguese", "ru": "Russian", "ur": "Urdu",
    "id": "Indonesian", "de": "German", "ja": "Japanese", "mr": "Marathi", "te": "Telugu",
    "tr": "Turkish", "ta": "Tamil", "yue": "Cantonese", "vi": "Vietnamese", "tl": "Tagalog",
    "ko": "Korean", "fa": "Persian", "ha": "Hausa", "sw": "Swahili", "jw": "Javanese",
    "it": "Italian", "pa": "Punjabi", "ms": "Malay", "gu": "Gujarati", "th": "Thai",
    "kn": "Kannada", "am": "Amharic", "yo": "Yoruba", "pl": "Polish", "ml": "Malayalam",
    "uk": "Ukrainian", "my": "Burmese", "su": "Sundanese", "uz": "Uzbek", "ln": "Lingala",
    "ro": "Romanian", "sd": "Sindhi", "nl": "Dutch", "ps": "Pashto", "ne": "Nepali",
    "mg": "Malagasy", "az": "Azerbaijani", "so": "Somali", "si": "Sinhala", "km": "Khmer",
    "as": "Assamese", "el": "Greek", "hu": "Hungarian", "sn": "Shona", "cs": "Czech",
    "sv": "Swedish", "ht": "Haitian Creole", "kk": "Kazakh", "sr": "Serbian", "be": "Belarusian",
    "bg": "Bulgarian", "he": "Hebrew", "tg": "Tajik", "da": "Danish", "fi": "Finnish",
    "sk": "Slovak", "no": "Norwegian", "hr": "Croatian", "ka": "Georgian", "ca": "Catalan",
    "hy": "Armenian", "sq": "Albanian", "lt": "Lithuanian", "tk": "Turkmen", "tt": "Tatar",
    "mn": "Mongolian", "lo": "Lao", "bs": "Bosnian", "gl": "Galician", "sl": "Slovenian",
    "lv": "Latvian", "mk": "Macedonian", "ba": "Bashkir", "et": "Estonian", "bo": "Tibetan",
    "yi": "Yiddish", "eu": "Basque", "cy": "Welsh", "af": "Afrikaans", "nn": "Norwegian Nynorsk",
    "lb": "Luxembourgish", "mt": "Maltese", "is": "Icelandic", "br": "Breton", "oc": "Occitan",
    "mi": "Maori", "fo": "Faroese", "haw": "Hawaiian", "la": "Latin", "sa": "Sanskrit",
}


def toml_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, dict):  # inline table
        return "{ " + ", ".join(f"{toml_value(k)} = {toml_value(x)}" for k, x in v.items()) + " }"
    # JSON escapes characters outside the BMP (emoji) as surrogate pairs, which TOML rejects.
    return json.dumps(v, ensure_ascii=False).replace("\x7f", "\\u007F")


def save_settings(s: dict[str, object]) -> None:
    """Write only the given settings, so unchanged ones keep following the environment."""
    os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
    try:
        with open(CONFIG, "rb") as f:
            tomllib.load(f)
    except tomllib.TOMLDecodeError:  # keep what can't be read, instead of overwriting it
        os.replace(CONFIG, f"{CONFIG}.unreadable-{time.strftime('%Y%m%d-%H%M%S')}")
    except FileNotFoundError:
        pass
    atomic_write(CONFIG, "# Written by the settings menu; overrides the ASR_* environment variables.\n"
                 + "".join(f"{k} = {toml_value(v)}\n" for k, v in s.items()))


def update_settings(changes: dict[str, object]) -> None:
    """Save only these settings on top of the current file, so a change made elsewhere in the
    meantime (e.g. a speech model download finishing) isn't overwritten with an old value."""
    if changes:
        with file_lock(CONFIG):
            save_settings({**read_config(), **changes})


def pending_model_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(runtime, f"{APP}-pending-model") if runtime else None


def pending_model() -> str | None:
    """The speech model being downloaded to switch to, if any."""
    try:
        with open(pending_model_path()) as f:
            return f.read().strip() or None
    except (FileNotFoundError, TypeError):
        return None


def set_pending_model(name: str | None) -> None:
    path = pending_model_path()
    if path:
        with open(path, "w") as f:
            f.write(name or "")


def fix_recording_text(name: str, text: str) -> None:
    """Replace the transcript of recordings/name in metadata.csv."""
    meta = os.path.join(RECORDINGS, "metadata.csv")
    with file_lock(meta):  # F4 appends to the same file
        with open(meta, newline="") as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:
            if row and row[0] == name:
                row[1] = text
        out = io.StringIO()
        csv.writer(out).writerows(rows)
        atomic_write(meta, out.getvalue())


def command_path() -> str:
    """How shortcuts, the service and the launcher run TuxWhisper: ./asr in a git clone (the
    wrapper sets TUXWHISPER_CMD), else the `tuxwhisper` command that pip or uv installed."""
    return os.environ.get("TUXWHISPER_CMD") or shutil.which(APP) or os.path.join(OLD_REPO, "asr")
