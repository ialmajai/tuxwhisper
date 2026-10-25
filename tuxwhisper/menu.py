"""The settings menu, the speech model picker and correcting the last transcript."""
import dataclasses
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

from tuxwhisper import daemon, desktop, engine, llm, store


def keys_menu():
    """Change the key for each action; GNOME applies it immediately."""
    def esc(text):  # zenity reads --text as Pango markup
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    while True:
        shortcuts = desktop.gnome_shortcuts()
        if shortcuts is None:
            desktop.zenity("--info", "--title=Keys", "--text=On this desktop, keys are set in its own shortcut "
                   "settings. See docs/other-desktops.md for each desktop.")
            return
        ours = {cmd: desktop.our_shortcut(shortcuts, cmd) for cmd in desktop.KEY_ACTIONS}
        rows = [x for cmd, label in desktop.KEY_ACTIONS.items()
                for x in (cmd, label, desktop.key_label(ours[cmd][1]) if ours[cmd] else "not bound")]
        cmd = desktop.zenity("--list", "--title=Keys", *desktop.fit(("Action", "Key"), zip(rows[1::3], rows[2::3])),
                     "--text=Pick an action to change its key.", "--column=cmd", "--column=Action",
                     "--column=Key", "--hide-column=1", "--print-column=1",
                     "--ok-label=Change", "--cancel-label=Back", *rows)
        if not cmd:
            return
        cmd = cmd.split("|")[0]
        path = ours[cmd][0] if ours[cmd] else None
        picked = desktop.pick(f"Key: {desktop.KEY_ACTIONS[cmd]}", [*desktop.KEY_SUGGESTIONS, "Other…"],
                      desktop.key_label(ours[cmd][1]) if ours[cmd] else None)
        if picked == "Other…":
            picked = desktop.zenity("--entry", f"--title=Key: {desktop.KEY_ACTIONS[cmd]}",
                            "--text=Type a key or combination, e.g. F6, Ctrl+Alt+D or Super+Space:")
        if not picked:
            continue
        binding = desktop.parse_key(picked)
        if not binding:
            desktop.zenity("--error", "--title=Keys", f"--text={esc(picked)} isn't a key TuxWhisper can use. "
                   "Try a function key (F1–F24) or a combination with Ctrl, Alt, Shift or Super.")
            continue
        other = next((c for p, (b, c) in shortcuts.items() if b == binding and p != path), None)
        if other is not None and desktop.zenity(
                "--question", "--title=Keys",
                f"--text={esc(desktop.key_label(binding))} already runs:\n{esc(other)}\n\nUse it anyway?") is None:
            continue
        desktop.set_shortcut(cmd, binding, path, list(shortcuts))


def with_service_env():
    """Re-run with the daemon's Environment= lines, so the menu shows the daemon's defaults."""
    if os.environ.get("TUXWHISPER_SERVICE_ENV"):
        return
    try:
        r = subprocess.run(["systemctl", "--user", "show", store.APP, "-p", "Environment", "--value"],
                           capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):  # no systemd: use our own environment
        return
    env = dict(os.environ, TUXWHISPER_SERVICE_ENV="1")
    env.update(item.split("=", 1) for item in shlex.split(r.stdout) if "=" in item)
    os.execve(sys.executable, [sys.executable, "-m", store.APP, "settings"], env)


def switch_asr_model(name):
    """Save the speech model and restart the daemon with it (about 15 s); True if it restarted."""
    store.update_settings({"asr_model": name})
    if subprocess.run(["systemctl", "--user", "is-active", "-q", store.APP]).returncode:
        return False
    subprocess.run(["systemctl", "--user", "restart", store.APP])
    return True


def download_asr_model(name):
    """`tuxwhisper download`: fetch a speech model while the daemon keeps dictating with the
    current one, then switch to it, unless another model was picked in the meantime."""
    size = next((m[2] for m in engine.ASR_MODELS if m[0] == name), "")
    nid = desktop.notify(f"Downloading the speech model {name}…", persistent=True,
                         body="<i>" + (f"About {size}. " if size else "")
                         + "Keep dictating; TuxWhisper switches when it's done.</i>")
    try:
        if engine.DEVICE == "vulkan":
            engine.download_ggml(name)
        else:
            from faster_whisper.utils import download_model
            download_model(name)
    except Exception as e:  # noqa: BLE001 - any failure: say so and keep the current model
        desktop.close_notification(nid)
        if store.pending_model() == name:
            store.set_pending_model(None)
        desktop.notify(f"Couldn't download the speech model {name}", body="Still using the current one.",
                       level="error")
        sys.exit(f"download of {name} failed: {e}")
    desktop.close_notification(nid)
    if store.pending_model() != name:  # the user switched to another model while this downloaded
        return
    store.set_pending_model(None)
    if switch_asr_model(name):
        desktop.notify(f"Downloaded {name}", body="Switching to it now (about 15 seconds).")
    else:
        desktop.notify(f"Downloaded {name}", body="It's used from the next start of TuxWhisper.")


def pick_asr_model(current):
    """Choose the speech model. A downloaded one is used at once; a new one downloads in the
    background while dictation carries on with the current model."""
    current = current or engine.short_name(engine.default_asr_model())
    pending = store.pending_model()
    def state(n):
        return ", downloading" if n == pending else ", downloaded" if engine.model_downloaded(n) else ""
    labels = {f"{n} ({size}{state(n)}): {about}": n for n, about, size in engine.asr_models()}
    picked = desktop.zenity("--list", "--title=Speech model", *desktop.fit(("Current", "Model"), [("✓", l) for l in labels]),
                    "--text=The model that turns your speech into text. Bigger is more accurate but slower.",
                    "--column=Current", "--column=Model", "--print-column=2",
                    *[x for label, n in labels.items() for x in ("✓" if n == current else "", label)])
    if not picked or labels[picked] in (current, pending):
        return
    name = labels[picked]
    if engine.model_downloaded(name):
        store.set_pending_model(None)  # a download in progress finishes, but won't switch away from this
        restarted = switch_asr_model(name)
        desktop.zenity("--info", "--title=Speech model", f"--text=Switching to {name}. "
               + ("Dictation is back in about 15 seconds." if restarted
                  else "It's used from the next start of TuxWhisper."))
        return
    store.set_pending_model(name)
    subprocess.Popen([sys.executable, "-m", store.APP, "download", name], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    desktop.zenity("--info", "--title=Speech model", f"--text=Downloading {name} in the background. Keep "
           f"dictating with {current}; TuxWhisper switches to {name} when the download is done.")


def menu_choice(rows):
    """Show the settings list (rows: (key, setting, value, help)) and return the picked key or
    None, with each row's help in a column of its own."""
    text = "--text=Pick a row to change it. Most changes apply to your next dictation."
    if desktop.have_yad():
        # No hidden key column here, so rows are told apart by their visible names.
        by_name = {name: key for key, name, _, _ in rows}
        r = subprocess.run(["yad", *desktop.YAD_APP, "--list", "--title=TuxWhisper Settings",
                            *desktop.fit(("Setting", "Value", "What it does"), [row[1:] for row in rows]),
                            "--center", text, "--column=Setting", "--column=Value", "--column=What it does",
                            "--print-column=1",
                            "--button=Close:1", "--button=Select:0", "--",
                            *[x for row in rows for x in row[1:]]],
                           capture_output=True, text=True)
        name = r.stdout.strip().split("|")[0]
        return by_name.get(name) if r.returncode == 0 else None
    choice = desktop.zenity("--list", "--title=TuxWhisper Settings",
                    *desktop.fit(("Setting", "Value", "What it does"), [row[1:] for row in rows]), text,
                    "--column=key", "--column=Setting", "--column=Value", "--column=What it does",
                    "--hide-column=1", "--print-column=1", "--ok-label=Select", "--cancel-label=Close",
                    *[x for row in rows for x in row])
    return choice.split("|")[0] if choice else None


def unload_label(minutes, never):
    """Menu wording for an idle-unload setting; never is the value that means "keep loaded"."""
    if minutes == never or minutes < 0:
        return "always loaded"
    return "unload right after use" if minutes == 0 else f"unload after {minutes} min idle"


# The paste keys as people write them; the settings file keeps store.PASTE_KEYS' names.
PASTE_LABELS = {"shift+insert": "Shift+Insert", "ctrl+v": "Ctrl+V", "ctrl+shift+v": "Ctrl+Shift+V"}


def keys_summary():
    shortcuts = desktop.gnome_shortcuts()
    if shortcuts is None:
        return "set in your desktop"
    keys = [desktop.key_label(ours[1]) if (ours := desktop.our_shortcut(shortcuts, cmd)) else "–" for cmd in desktop.KEY_ACTIONS]
    return ", ".join(keys)


@dataclasses.dataclass
class Setting:
    """One row of the settings menu. edit asks for a new value and returns the settings to save,
    or None when there's nothing to save (cancelled, or an action such as a dialog of its own);
    after runs once they're saved."""
    key: str  # a config.toml key, whose row says "(default)" until it's set; else just an ID
    label: str  # {mode} is the current cleanup style
    value: Callable[[store.Settings], str]
    edit: Callable[[store.Settings], dict | None]
    help: str
    after: Callable[[], None] | None = None


def edit_language(s):
    names = {"auto-detect": "", **{f"{n} ({c})": c for c, n in store.LANGUAGES.items()}}
    current = next((n for n, c in names.items() if c == s.language), "auto-detect")
    picked = desktop.pick("Language", list(names), current)
    return {"language": names[picked]} if picked else None


def edit_paste_key(s):
    labels = {PASTE_LABELS[k]: k for k in store.PASTE_KEYS}
    picked = desktop.pick("Paste key", list(labels), PASTE_LABELS[s.paste_key])
    return {"paste_key": labels[picked]} if picked else None


def edit_unload(key, title, never):
    """An idle-unload setting's editor; never is the value that means "keep loaded"."""
    options = [-1, 5, 10, 30, 60] if never == 0 else [0, 5, 10, 30, 60, -1]
    def edit(s):
        labels = {unload_label(m, never): m for m in options}
        picked = desktop.pick(title, list(labels), unload_label(getattr(s, key), never))
        return {key: max(labels[picked], never)} if picked else None  # "always loaded" is stored as never
    return edit


def edit_sound_volume(s):
    picked = desktop.zenity("--scale", "--title=Sound volume", "--min-value=0", "--max-value=100",
                            "--step=5", f"--value={s.sound_volume}", "--text=Percent of your system volume:")
    return {"sound_volume": int(picked)} if picked else None


def edit_mode(s):
    picked = desktop.pick("Cleanup style", list(llm.modes()), llm.current_mode())
    return {"mode": picked} if picked else None


def ollama_models():
    """The installed Ollama models, so cleanup can't be pointed at one that doesn't exist; None
    (after saying why) when there are none to offer."""
    try:
        models = sorted(m["name"] for m in llm.ollama("/api/tags", timeout=3)["models"])
    except (OSError, ValueError, KeyError):
        desktop.zenity("--error", "--title=Cleanup model",
                       f"--text=Can't reach Ollama at {llm.OLLAMA_URL}. Start it and try again.")
        return None
    if not models:
        desktop.zenity("--info", "--title=Cleanup model",
                       "--text=No Ollama models installed. Install one with: ollama pull llama3.2")
        return None
    return models


def edit_rewrite_model(s):
    models = ollama_models()
    if not models:
        return None
    current = s.rewrite_model
    picked = desktop.pick("Cleanup model", models, current if ":" in current else current + ":latest")
    return {"rewrite_model": picked} if picked else None


def edit_mode_model(s):
    models = ollama_models()
    if not models:
        return None
    mode, same = llm.current_mode(), "same as cleanup model"
    picked = desktop.pick(f"Model for {mode}", [same, *models], s.mode_models.get(mode, same))
    if not picked:
        return None
    mode_models = {k: v for k, v in s.mode_models.items() if k != mode}
    if picked != same:
        mode_models[mode] = picked
    return {"mode_models": mode_models}


def show_history(s):
    """Recent transcripts: pick one to copy it again. Saves nothing."""
    items = store.load_history()[::-1]  # newest first
    if not items:
        desktop.zenity("--info", "--title=Recent transcripts", "--text=Nothing dictated this session yet.")
        return
    rows = [x for i, h in enumerate(items)
            for x in (str(i), h["time"], h["text"].replace("\n", " ⏎ ")[:90])]
    picked = desktop.zenity("--list", "--title=Recent transcripts", *desktop.fit(("Time", "Text"), zip(rows[1::3], rows[2::3])),
                    "--text=Pick one to copy it to the clipboard.",
                    "--column=i", "--column=Time", "--column=Text",
                    "--hide-column=1", "--print-column=1", "--ok-label=Copy", *rows)
    if picked:
        desktop.clipboard_set(items[int(picked.split("|")[0])]["text"])
        desktop.notify("Copied", body="Paste it with Ctrl+V.")


def action(fn):
    """A row that runs fn (a dialog of its own) and saves nothing here."""
    def run(s):
        fn()
    return run


def edit_file(path, title, header):
    return action(lambda: desktop.edit_list(path, title, header))


# In groups: past transcripts, speech, LLM cleanup, then keys and sounds.
SETTINGS = [
    Setting("correct", "Correct last transcript", lambda s: "Fix…", action(lambda: correct_last()),
            "Fix the last transcript; TuxWhisper offers to remember the fix for next time"),
    Setting("history", "Recent transcripts", lambda s: f"{len(store.load_history())} this session", show_history,
            "Your last 20 transcripts; pick one to copy it again"),
    Setting("language", "Language", lambda s: store.LANGUAGES.get(s.language, "auto-detect"), edit_language,
            "The language you speak; fixed avoids misdetection on short clips"),
    Setting("asr_model", "Speech model", lambda s: s.asr_model or engine.short_name(engine.default_asr_model()),
            lambda s: pick_asr_model(s.asr_model),  # returns None: it saves and restarts itself
            "The Whisper model: bigger is more accurate, smaller is faster"),
    Setting("asr_unload_minutes", "Speech model in memory", lambda s: unload_label(s.asr_unload_minutes, 0),
            edit_unload("asr_unload_minutes", "Speech model in memory", 0),
            "Free GPU memory when you haven't dictated for a while; reloads in under a second"),
    Setting("vocab", "Vocabulary", lambda s: "Edit…", edit_file(store.VOCAB, "Vocabulary", store.VOCAB_HEADER),
            "Names and jargon Whisper should spell right (English)"),
    Setting("replacements", "Replacements", lambda s: "Edit…",
            edit_file(store.REPLACEMENTS, "Replacements", store.REPLACEMENTS_HEADER),
            'Spoken words to replace, e.g. "new line" with a line break'),
    Setting("mode", "Cleanup style", lambda s: llm.current_mode(), edit_mode,
            "What LLM cleanup does: clean up, fix grammar, email, bullet points…"),
    Setting("rewrite_model", "Cleanup model", lambda s: s.rewrite_model, edit_rewrite_model,
            "The Ollama model used for LLM cleanup"),
    Setting("mode_model", "Model for {mode}", lambda s: s.mode_models.get(llm.current_mode(), "same as cleanup model"),
            edit_mode_model, "A different LLM model for this cleanup style only"),
    Setting("llm_unload_minutes", "LLM in memory", lambda s: unload_label(s.llm_unload_minutes, -1),
            edit_unload("llm_unload_minutes", "LLM in memory", -1),
            "Free the cleanup model's GPU memory after a while; reloading takes a few seconds"),
    Setting("modes", "Custom cleanup styles", lambda s: "Edit…",
            edit_file(store.MODES_FILE, "Custom cleanup styles",
                      "# One cleanup style per line: Name => instruction for the LLM, e.g.\n"
                      "# Tweet => Rewrite as a tweet under 280 characters.\n"),
            "Your own cleanup styles, as Name => instruction"),
    Setting("keys", "Keys", lambda s: keys_summary(), action(lambda: keys_menu()),
            "The keys for dictation, cleanup, saving and this menu"),
    Setting("paste_key", "Paste key", lambda s: PASTE_LABELS[s.paste_key], edit_paste_key,
            "Key used to paste; try Ctrl+V if an app doesn't paste"),
    Setting("sounds", "Sounds", lambda s: "On" if s.sounds else "Off", lambda s: {"sounds": not s.sounds},
            "Beep when recording starts and stops"),
    Setting("sound_volume", "Sound volume", lambda s: f"{s.sound_volume}%", edit_sound_volume,
            "How loud the beeps are", after=lambda: desktop.sound("device-added")),  # a preview
]


def settings_menu():
    if not shutil.which("zenity"):
        sys.exit("The settings menu needs zenity (e.g. sudo apt install zenity).")
    fields = {f.name for f in dataclasses.fields(store.Settings)}
    while True:
        s, cfg, mode = store.settings(), store.read_config(), llm.current_mode()
        # The menu can't see the service's environment, so unchanged settings say so.
        rows = [(row.key, row.label.replace("{mode}", mode),
                 row.value(s) + (" (default)" if row.key in fields and row.key not in cfg else ""), row.help)
                for row in SETTINGS]
        choice = menu_choice(rows)
        if not choice:
            return
        row = next(r for r in SETTINGS if r.key == choice)
        changes = row.edit(s)
        if changes is not None:
            store.update_settings(changes)
            if row.after:
                row.after()



WORD_EDGES = ".,!?;:\"'()[]“”‘’"


def suggest_learning(before, after, vocabulary=()):
    """What a correction can teach, as (kind, spoken, written, ticked) with kind "rule" (a
    replacement) or "vocab". Only short changed phrases count: a longer edit is a rewrite, not
    a mishearing. Names and jargon are ticked; plain-word rules start unticked, because a
    one-off fix like "two" -> "three" would be wrong as a rule."""
    a = [w.strip(WORD_EDGES) for w in before.split()]
    b = [w.strip(WORD_EDGES) for w in after.split()]
    a, b = [w for w in a if w], [w for w in b if w]
    known = {v.lower() for v in vocabulary}
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "replace" or i2 - i1 > 4 or j2 - j1 > 4:
            continue
        old, new = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        special = any(c.isupper() or c.isdigit() for c in new)
        out.append(("rule", old.lower(), new, special))
        for i, w in enumerate(b[j1:j2], start=j1):
            if i > 0 and any(c.isupper() for c in w) and w.lower() not in known:
                known.add(w.lower())
                out.append(("vocab", w, w, True))
    return out


def append_line(path, line, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        f.write((header if new else "") + line + "\n")



RETYPE_WITHIN = 600  # seconds after a dictation during which a correction retypes it in place


def correct_last():
    """Edit the last transcript, copy the fix, and offer to learn from it."""
    items = store.load_history()
    if not items:
        desktop.zenity("--info", "--title=Correct last transcript", "--text=Nothing dictated this session yet.")
        return
    last = items[-1]
    # Erase the dictation (it was typed with a trailing space) and type the fix in its place,
    # but only if it was really typed and recently: after a while the cursor has likely moved.
    def fresh():
        return last.get("typed") and time.time() - last.get("at", 0) < RETYPE_WITHIN
    retype = fresh()
    fixed = desktop.edit_text(
        last["text"], "Correct last transcript", "Replace" if retype else "Copy",
        "Fix the words, then Replace: TuxWhisper retypes it where you dictated it." if retype
        else "Fix the words, then Copy, and paste the corrected text where you need it.")
    if fixed is None or fixed.strip() == last["text"].strip():
        return
    fixed, before = fixed.strip(), last["text"]
    replaced = False
    if retype and fresh():  # checked again: the dialog may have stayed open a long time
        with open(store.retype_path(), "w") as f:
            json.dump({"delete": daemon.typed_length(before) + 1, "text": fixed, "at": last.get("at")}, f)
        replaced = daemon.request("retype") == "ok"
    if not replaced:  # hand the fix over to paste by hand
        desktop.clipboard_set(fixed)
    store.update_history_entry(last, text=fixed, typed=replaced, at=time.time())
    done = "Replaced the dictated text." if replaced else "The corrected text is copied. Paste it with Ctrl+V."

    choices = [(f'Always write "{spoken}" as "{written}"' if kind == "rule"
                else f'Add "{written}" to the vocabulary', kind, spoken, written, ticked)
               for kind, spoken, written, ticked in suggest_learning(before, fixed, (store.vocab() or "").split(", "))]
    if last.get("recording") and not last.get("rewritten"):
        choices.append(("Fix the saved recording's transcript (for fine-tuning)", "recording",
                        last["recording"], fixed, True))
    picked = set()
    if choices:
        rows = [x for i, c in enumerate(choices) for x in ("TRUE" if c[4] else "FALSE", str(i), c[0])]
        out = desktop.zenity("--list", "--checklist", "--title=Learn from this correction",
                     *desktop.fit(("", "Change"), [("", c[0]) for c in choices]),
                     f"--text={done}\nTick what TuxWhisper should remember:",
                     "--column=", "--column=i", "--column=Change", "--hide-column=2",
                     "--print-column=2", "--separator= ", "--ok-label=Remember",
                     "--cancel-label=Skip", *rows)
        picked = {int(i) for i in re.findall(r"\d+", out or "")}
    learned = []
    for i in sorted(picked):
        _, kind, spoken, written, _ = choices[i]
        if kind == "rule":
            append_line(store.REPLACEMENTS, f"{spoken} => {written}",
                        store.REPLACEMENTS_HEADER)
        elif kind == "vocab":
            append_line(store.VOCAB, written, store.VOCAB_HEADER)
        else:
            store.fix_recording_text(spoken, written)
        learned.append(kind)
    remembered = f"Remembered {desktop.plural(len(learned), 'change', 'changes')}." if learned else ""
    desktop.notify("Replaced the dictated text" if replaced else "Corrected text copied",
                   body=(remembered if replaced else f"Paste it with Ctrl+V. {remembered}").strip() or None)
