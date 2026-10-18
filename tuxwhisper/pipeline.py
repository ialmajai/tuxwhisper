"""One stopped take, from audio to text at the cursor: transcription, cleanup and pasting."""
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from tuxwhisper import desktop, engine, llm, store
from tuxwhisper.recordings import Recordings

log = logging.getLogger(__name__)

# Clipping: samples at full scale, as when speaking too close to the mic or with the gain too high.
CLIP_LEVEL, CLIP_SHARE, CLIP_WARN_EVERY = 0.99, 0.001, 600  # level, share of samples, seconds
COPY_FAILED = ("Couldn't copy the transcript", "Find it under Recent transcripts in the settings menu.")
MIC_ICON = "audio-input-microphone"


class Status(list):
    """A take's "Transcribing…" notification ID (one item once shown), and the thread showing it."""
    thread = None


def wait_shown(status):
    """Wait until the take's "Transcribing…" is shown, so it can be replaced or closed."""
    if getattr(status, "thread", None) is not None:
        status.thread.join()


def end_status(status):
    """Close a take's "Transcribing…" notification, once: status is a one-item list."""
    wait_shown(status)
    if status:
        desktop.close_notification(status.pop())


@dataclass
class TakeContext:
    """Everything one take needs, captured when its recording stops: the next take may start
    (in another mode) before this one is pasted."""
    chunks: list  # the mic's blocks of samples
    excess: int = 0  # samples recorded after the key's release, to drop
    rewrite: bool = False  # clean it up with the LLM
    warming: Future | None = None  # the cleanup model's warm-up, for a cleanup take
    ticket: int = 0  # its turn: takes are pasted in the order they stopped
    indicator: str | None = None  # the "Recording…" notification, which becomes "Transcribing…"
    status: Status = field(default_factory=Status)  # the "Transcribing…" notification

    def audio(self, np):
        audio = np.concatenate(self.chunks)[:, 0] if self.chunks else []
        return audio[:-self.excess] if self.excess else audio


class TakePipeline:
    """Turns a stopped take into text at the cursor: transcribe, tell the user when speech was
    missed, clean up, apply replacements, paste, then note it in the history."""

    def __init__(self, models: engine.ModelManager, paster: "Paster", recordings: Recordings,
                 after_paste: Callable[[], None] = lambda: None):
        import numpy as np

        self.np = np
        self.models, self.paster, self.recordings = models, paster, recordings
        self.after_paste = after_paste
        self.last_clip_warning = -CLIP_WARN_EVERY

    def run(self, take: TakeContext) -> None:
        """Transcribe one take, clean it up if asked, and paste it. take.status holds its
        "Transcribing…" notification, closed once there's a result or a notice to show."""
        audio, status = take.audio(self.np), take.status
        if len(audio) < store.RATE * 0.3:  # too short to be speech; Whisper hallucinates on these
            return
        # Read the clipboard to restore after the paste while the take transcribes.
        snapshot = self.paster.snapshot_clipboard() if self.paster.kbd is not None else None
        if not self.models.ensure_loaded():
            return
        t0 = time.time()
        result = self.models.transcribe(audio, store.settings())
        if result is None:
            return
        text, lang = result
        self.models.touch()
        log.info(f"[{len(audio) / store.RATE:.1f}s audio (-{take.excess / store.RATE:.2f}s tail), "
                 f"{time.time() - t0:.2f}s asr] {text!r}")
        missed = missed_speech(len(audio) / store.RATE, text)
        if missed:
            log.info(f"  {missed[0]}: {missed[1]}")
            end_status(status)
            desktop.notify(missed[0], body=missed[1], level="warn")
        if not text:
            return
        kept = self.recordings.keep(audio, text)  # saved takes keep Whisper's text, matching the audio
        if take.rewrite:
            text = self.clean_up(text, lang, take.warming, status)
        text = store.apply_replacements(text)
        end_status(status)
        typed = self.paster.paste(text + " ", snapshot)
        kept.entry = store.add_history(text, rewritten=take.rewrite, typed=typed)  # only a typed take can be retyped
        self.warn_if_clipped(audio)
        self.after_paste()

    def clean_up(self, text, lang, warming, status):
        """The LLM's cleanup of text; the plain text if it fails (llm says so)."""
        t0 = time.time()
        # Say so only when the cleanup is slow (the model is still loading, usually), in place
        # of "Transcribing…": the same notification, so nothing flashes up and away.
        def cleaning():
            wait_shown(status)
            nid = desktop.notify(f"Cleaning up ({llm.current_mode()})…", persistent=True,
                                 icon=MIC_ICON, replace=status[0] if status else None)
            status[:] = [nid]
        timer = threading.Timer(0.8, cleaning)
        timer.start()
        on_gpu = None  # not known: llm_cleanup checks
        if warming:
            on_gpu = True if warming.result() is True else None  # a failed warm-up is checked again
        text = llm.llm_cleanup(text, lang, on_gpu=on_gpu)
        timer.cancel()
        timer.join()
        log.info(f"  [{time.time() - t0:.2f}s rewrite] {text!r}")
        return text

    def warn_if_clipped(self, audio):
        """Tell the user, after the paste, when the mic was too loud; at most every 10 minutes."""
        clipped = float((self.np.abs(audio) >= CLIP_LEVEL).mean())
        if clipped > CLIP_SHARE and time.monotonic() - self.last_clip_warning > CLIP_WARN_EVERY:
            self.last_clip_warning = time.monotonic()
            log.info(f"  clipping: {clipped:.2%} of samples")
            desktop.notify("Your mic is clipping", level="warn",
                           body="Move back a little, or lower its input volume in Settings → Sound.")


def missed_speech(seconds: float, text: str) -> tuple[str, str] | None:
    """(title, body) when Whisper heard far less than was recorded, else None. Its silence
    filter drops speech it can't make out (too quiet, whispered, a muted mic), and without a
    notice a long take would just vanish. Under a second is usually a stray key press."""
    hint = "Speak up, or check the mic and its input volume in Settings → Sound."
    words = len(text.split())
    if not words and seconds >= 1:
        return "Didn't catch that", f"No speech found in {seconds:.0f} s of audio. {hint}"
    if seconds >= 10 and words < seconds / 3:  # normal speech is 2–3 words a second
        return ("Only caught a few words",
                f"{desktop.plural(words, 'word', 'words')} from {seconds:.0f} s of audio; some may be missing. {hint}")
    return None


class Paster:
    """Puts text at the cursor through the clipboard and a paste key on the virtual keyboard,
    then gives the clipboard back its old contents."""

    def __init__(self, kbd: Any, ec: Any):
        """kbd: the virtual keyboard (None: copy instead of typing); ec: evdev's ecodes."""
        self.kbd, self.ec = kbd, ec

    def snapshot_clipboard(self):
        """Start reading the selections the paste key overwrites; paste() takes the result."""
        selections = paste_selections(store.settings().paste_key)
        saved = {}
        thread = threading.Thread(target=lambda: saved.update(read_selections(selections)), daemon=True)
        thread.start()
        return thread, saved

    def paste(self, text: str, snapshot=None) -> bool:
        """Type text at the cursor; True if it was typed, False if it was only copied (or not even that).
        snapshot is snapshot_clipboard()'s, taken before the take was transcribed."""
        # Clipboard + a paste shortcut handles any Unicode and works on X11 and Wayland.
        if self.kbd is None:
            if desktop.clipboard_set(text):
                desktop.notify("Copied", body="Paste it with Ctrl+V.")
            else:
                desktop.notify(*COPY_FAILED, level="error")
            return False
        paste_key = store.settings().paste_key
        keys = [getattr(self.ec, k) for k in store.PASTE_KEYS[paste_key]]
        selections = paste_selections(paste_key)
        saved = {}
        if snapshot:
            thread, saved = snapshot
            thread.join()
        # The paste key may have changed since the snapshot: read what it didn't.
        saved.update(read_selections([sel for sel in selections if sel not in saved]))
        saved = {sel: saved[sel] for sel in selections}
        if not all(write_selections({sel: text for sel in saved}).values()):  # pasting now would paste old text
            desktop.notify(*COPY_FAILED, level="error")
            return False
        time.sleep(0.05)
        self.press(keys)
        time.sleep(0.3)  # the app fetches the clipboard asynchronously after the paste
        write_selections({sel: old for sel, old in saved.items() if old is not None})
        return True

    def press(self, keys):
        """Press keys together, then release them in reverse order."""
        ec, k = self.ec, self.kbd
        presses = [(c, 1) for c in keys] + [(c, 0) for c in reversed(keys)]
        for code, val in presses:
            k.write(ec.EV_KEY, code, val)
            k.syn()
            time.sleep(0.01)

    def erase(self, count: int) -> None:
        """Press BackSpace count times."""
        k, ec = self.kbd, self.ec
        for _ in range(count):
            for val in (1, 0):
                k.write(ec.EV_KEY, ec.KEY_BACKSPACE, val)
                k.syn()
            time.sleep(0.004)


def paste_selections(paste_key):
    """The selections a paste key pastes from. Some apps (e.g. terminals) paste the primary
    selection on Shift+Insert, so set both; the Ctrl shortcuts always paste the clipboard."""
    return ("clipboard", "primary") if paste_key == "shift+insert" else ("clipboard",)


def in_parallel(fn, items):
    """{item: fn(item)}, each call in its own thread: the clipboard tools take tens of ms each."""
    out = {}
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, fn(i)), daemon=True) for i in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def read_selections(selections):
    return in_parallel(desktop.clipboard_get, selections)


def write_selections(texts):
    """Set each selection to its text; {selection: True if it was set}."""
    return in_parallel(lambda sel: desktop.clipboard_set(texts[sel], sel), texts)
