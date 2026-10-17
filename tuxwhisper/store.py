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
import wave

APP = "tuxwhisper"
LANGUAGE = os.environ.get("ASR_LANGUAGE") or None  # e.g. "en"; None = auto-detect
REWRITE_MODEL = os.environ.get("ASR_REWRITE_MODEL", "llama3.2")
# Shift+Insert pastes in browsers, editors and terminals alike, and unlike Ctrl+V it doesn't
# depend on the keyboard layout (keys are sent by position, so Ctrl+V is Ctrl+K on Dvorak).
PASTE_KEY = os.environ.get("ASR_PASTE_KEY", "shift+insert")
SOUNDS = os.environ.get("ASR_SOUNDS", "1") != "0"  # beep on start and stop
# Percent of the system volume, so the beeps never get louder than your other sounds.
SOUND_VOLUME = os.environ.get("ASR_SOUND_VOLUME", "40")  # validated in settings()
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


def atomic_write(path, text):
    """Replace path's contents in one step: a reader never sees a half-written file, and a
    crash mid-write leaves the old file instead of a truncated one."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_config():
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


def settings():
    """Live settings: environment defaults overridden by the config file, re-read on every use."""
    s = {"sounds": SOUNDS, "sound_volume": SOUND_VOLUME, "language": LANGUAGE or "",
         "paste_key": PASTE_KEY, "rewrite_model": REWRITE_MODEL, "mode": "Clean up",
         "asr_model": os.environ.get("ASR_MODEL", ""),
         # Minutes idle before freeing GPU memory: speech model 0 = never, LLM -1 = never.
         "asr_unload_minutes": 30, "llm_unload_minutes": 30,
         "mode_models": {}}
    s.update({k: v for k, v in read_config().items() if k in s})
    try:
        s["sound_volume"] = min(max(int(s["sound_volume"]), 0), 100)
    except (TypeError, ValueError):
        s["sound_volume"] = 40
    if s["paste_key"] not in PASTE_KEYS:
        s["paste_key"] = "shift+insert"
    if not isinstance(s["mode_models"], dict):
        s["mode_models"] = {}
    for key, low in (("asr_unload_minutes", 0), ("llm_unload_minutes", -1)):
        if not isinstance(s[key], int) or isinstance(s[key], bool) or s[key] < low:
            s[key] = 30
    if s["language"] not in LANGUAGES:  # also catches a wrong type from a hand-edited config
        s["language"] = ""
    return s


MODES_FILE = os.path.join(CONFIG_DIR, "modes.txt")


def vocab():
    try:
        with open(VOCAB) as f:
            words = [w.strip() for w in f if w.strip() and not w.startswith("#")]
    except FileNotFoundError:
        return None
    return ", ".join(words) or None


def apply_replacements(text):
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


def dataset_size(meta):
    """Number of saved takes and their total length in seconds."""
    takes, frames = 0, 0
    with open(meta, newline="") as f:
        for row in csv.DictReader(f):
            try:
                with wave.open(os.path.join(RECORDINGS, row["file_name"])) as w:
                    frames += w.getnframes()
                takes += 1
            except (OSError, KeyError, wave.Error):  # file deleted by hand
                pass
    return takes, frames / RATE


def history_path():
    # The runtime dir is private to the user and emptied at logout, so dictated text
    # doesn't pile up on disk.
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(runtime, f"{APP}-history.json") if runtime else None


def load_history():
    try:
        with open(history_path()) as f:
            return json.load(f)
    except (TypeError, OSError, ValueError):
        return []


def add_history(text, rewritten=False, typed=False):
    """Keep the last 20 transcripts for "Recent transcripts" and "Correct last transcript".
    typed: it was typed at the cursor (not just copied), so a correction may erase and retype it."""
    if history_path():
        entry = {"time": time.strftime("%H:%M"), "text": text, "rewritten": rewritten,
                 "typed": typed, "at": time.time()}
        with file_lock(history_path()):
            save_history((load_history() + [entry])[-20:])
        return entry
    return None


def save_history(items):
    atomic_write(history_path(), json.dumps(items))


def update_history_entry(entry, **changes):
    """Change one history entry, found again by its time and text: other takes may have been
    added since it was read."""
    with file_lock(history_path()):
        items = load_history()
        match = next((i for i in items if i.get("at") == entry.get("at") and i["text"] == entry["text"]), None)
        if match is not None:
            match.update(changes)
            save_history(items)


def mark_last_saved(name):
    """Note that F4 saved the last take as recordings/name, so a correction can fix its transcript."""
    items = load_history()
    if items:
        update_history_entry(items[-1], recording=name)


def retype_path():
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


def save_settings(s):
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


def update_settings(changes):
    """Save only these settings on top of the current file, so a change made elsewhere in the
    meantime (e.g. a speech model download finishing) isn't overwritten with an old value."""
    if changes:
        with file_lock(CONFIG):
            save_settings({**read_config(), **changes})


def pending_model_path():
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    return os.path.join(runtime, f"{APP}-pending-model") if runtime else None


def pending_model():
    """The speech model being downloaded to switch to, if any."""
    try:
        with open(pending_model_path()) as f:
            return f.read().strip() or None
    except (FileNotFoundError, TypeError):
        return None


def set_pending_model(name):
    path = pending_model_path()
    if path:
        with open(path, "w") as f:
            f.write(name or "")


def fix_recording_text(name, text):
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


def command_path():
    """How shortcuts, the service and the launcher run TuxWhisper: ./asr in a git clone (the
    wrapper sets TUXWHISPER_CMD), else the `tuxwhisper` command that pip or uv installed."""
    return os.environ.get("TUXWHISPER_CMD") or shutil.which(APP) or os.path.join(OLD_REPO, "asr")
