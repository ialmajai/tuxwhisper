"""The dictation daemon and the socket the key commands reach it through."""
import csv
import datetime
import errno
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tuxwhisper import desktop, engine, llm, store
from tuxwhisper.client import request, send, sock_path  # noqa: F401 - re-exported for menu and tests

# Clipping: samples at full scale, as when speaking too close to the mic or with the gain too high.
CLIP_LEVEL, CLIP_SHARE, CLIP_WARN_EVERY = 0.99, 0.001, 600  # level, share of samples, seconds
HINT_EVERY = 300  # seconds between lookups of the settings key shown under "Recording…"


@dataclass
class Take:
    """A transcribed take, kept for the save key: its audio, Whisper's text (matching the
    audio), its history entry once pasted, and its number among the daemon's takes."""
    audio: Any
    text: str
    entry: dict | None = None
    id: int = 0


class Dictation:
    @classmethod
    def create(cls):
        """Load the speech model and open the virtual keyboard, then start dictating."""
        import sounddevice as sd
        from evdev import UInput, ecodes
        from evdev.uinput import UInputError

        models = engine.ModelManager.load()  # warmed up; no model yet if no GPU memory
        # Register every paste key so the menu can switch between them without a restart.
        all_keys = sorted({getattr(ecodes, k) for keys in store.PASTE_KEYS.values() for k in keys}
                          | {ecodes.KEY_BACKSPACE})  # erases a dictation when it's corrected
        try:
            kbd = UInput({ecodes.EV_KEY: all_keys}, name=f"{store.APP}-keyboard")
        except (UInputError, OSError) as e:  # no udev rule for /dev/uinput: copy instead of typing
            kbd = None
            print(f"virtual keyboard unavailable ({e}); transcripts go to the clipboard", flush=True)
            desktop.notify("No virtual keyboard", body="Transcripts are copied instead. Paste them with Ctrl+V.",
                           level="warn")
        d = cls(models, sd, kbd, ecodes)
        d.refresh_hint()
        d.REPEAT_WINDOW = repeat_window()
        threading.Thread(target=d._watch_release, daemon=True).start()
        threading.Thread(target=d._watch_idle, daemon=True).start()
        print("ready", flush=True)
        return d

    def __init__(self, models, sd, kbd, ec):
        """Dictation's state around the speech model (an engine.ModelManager), the sounddevice
        module, the virtual keyboard (None: copy instead of typing) and evdev's ecodes. create()
        passes the real ones and starts the watcher threads; tests pass fakes."""
        import numpy as np

        self.np, self.sd, self.ec = np, sd, ec
        self.models = models
        self.kbd = kbd
        self.lock = threading.Lock()
        self.press_lock = threading.Lock()  # presses come from one thread per connection
        # Takes are transcribed and pasted one at a time, in the order they stopped.
        self.turn = threading.Condition()
        self.next_ticket = self.serving = 0
        self.stream = None
        self.indicator = None  # ID of the on-screen recording notification
        self.hint = None  # the settings key shown under "Recording…"; see refresh_hint()
        self.last_hint = -HINT_EVERY
        self.last_clip_warning = -CLIP_WARN_EVERY
        self.chunks = []
        self.last_press = 0.0
        self.last_used = time.monotonic()
        self.last_take: Take | None = None  # the latest transcription, until the save key takes it
        self.takes = 0  # counts takes, so a cancelled save can tell whether a newer one came
        self.last_save = 0.0
        self.held = False
        self.rewrite = False  # the current take's mode, set by the key that started it
        # The cleanup model's warm-up for the current take, if it's a cleanup one.
        self.warming: Future | None = None
        self.background = ThreadPoolExecutor(max_workers=2, thread_name_prefix=store.APP)

    # GNOME re-runs the shortcut on key auto-repeat (~every 0.1s here, after a 0.5s delay),
    # so a burst of presses means the key is held: treat it as push-to-talk.
    REPEAT_WINDOW = 0.7   # press this soon after the previous one = auto-repeat (set from GNOME)
    RELEASE_GAP = 0.35    # held key with no repeat for this long = released
    RELEASE_TAIL = 0.15   # audio kept after the last repeat (release is within ~0.11s of it)

    def press(self, rewrite=False):
        with self.press_lock:  # two presses at once must not both count as the first
            now = time.monotonic()
            is_repeat = now - self.last_press < self.REPEAT_WINDOW
            self.last_press = now
            if is_repeat:
                self.held = self.stream is not None
                return
            self.held = False
        threading.Thread(target=self.toggle, kwargs={"rewrite": rewrite}, daemon=True).start()

    def _watch_release(self):
        while True:
            time.sleep(0.05)
            if self.held and time.monotonic() - self.last_press > self.RELEASE_GAP:
                self.held = False
                cut_at = self.last_press + self.RELEASE_TAIL
                threading.Thread(target=self.toggle, args=(cut_at,), daemon=True).start()

    def _watch_idle(self):
        """Free the speech model's GPU memory after the idle time set in the menu."""
        while True:
            time.sleep(15)
            self.unload_if_idle()
            self.restore_preferred_model()

    def restore_preferred_model(self):
        """Go back to the chosen model when there's room, between takes rather than during one."""
        if self.stream is None:
            self.models.restore_if_room()

    def unload_if_idle(self):
        if self.stream is None:
            self.models.unload_if_idle(store.settings().asr_unload_minutes, time.monotonic() - self.last_used)

    def _start(self, rewrite):
        """Open the mic for a new take (under self.lock); the stream, or None if the mic failed."""
        self.last_used = time.monotonic()
        threading.Thread(target=self.models.ensure_loaded, daemon=True).start()  # overlaps the speech
        self.chunks = []
        self.rewrite = rewrite  # mode is set by the key that started the take
        # A cold load takes seconds, so overlap it with the speech; the cleanup skips its own
        # GPU check when the warm-up confirmed it.
        self.warming = self.background.submit(llm.warm_rewrite_model) if rewrite else None
        desktop.sound("device-added")  # 0.2 s; over before the mic finishes opening
        try:
            stream = self.sd.InputStream(
                samplerate=store.RATE, channels=1, dtype="float32",
                callback=lambda data, *_: self.chunks.append(data.copy()))
            stream.start()
        except Exception as e:  # noqa: BLE001 - unplugged, busy or no mic: stay ready
            print(f"microphone unavailable: {e}", flush=True)
            desktop.notify("Microphone unavailable", body="Check that it's connected and not in use.",
                           level="error")
            return None
        self.stream, self.indicator = stream, None
        return stream

    def _announce(self, stream, rewrite):
        """Show "Recording…" outside self.lock: notify-send -p can take a while, and holding the
        lock would delay a quick stop. A stop that came first closes the notification."""
        nid = desktop.notify(f"Recording ({llm.current_mode()})…" if rewrite else "Recording…",
                             persistent=True, body=self.hint, icon=MIC_ICON)
        with self.lock:
            if self.stream is stream:
                self.indicator = nid
                return
        desktop.close_notification(nid)

    def _stop(self, cut_at):
        """Close the mic (under self.lock); the take as (chunks, excess, rewrite, warming, ticket,
        the recording notification's ID, which becomes "Transcribing…")."""
        # Drop the audio recorded between the key release (cut_at) and now.
        excess = int(max(0.0, time.monotonic() - cut_at) * store.RATE) if cut_at else 0
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:  # noqa: BLE001 - e.g. the mic was unplugged mid-take
            print(f"microphone error on stop: {e}", flush=True)
        finally:
            self.stream = None
        indicator, self.indicator = self.indicator, None
        desktop.sound("device-removed")
        chunks, self.chunks = self.chunks, []
        # This take's own mode: the next take may start before this one is pasted.
        ticket, self.next_ticket = self.next_ticket, self.next_ticket + 1
        return chunks, excess, self.rewrite, self.warming, ticket, indicator

    def toggle(self, cut_at=None, rewrite=False):
        with self.lock:
            if self.stream is None:
                stream = self._start(rewrite)
                take = None
            else:
                take = self._stop(cut_at)
        if take is None:  # started (or the mic failed)
            if stream is not None:
                self._announce(stream, rewrite)
            return
        chunks, excess, rewrite, warming, ticket, indicator = take
        # Swap "Recording…" for "Transcribing…" in place: one notification, not a new popup.
        # Shown in the background: notify-send -p takes a while, and the take needn't wait.
        status = Status()
        status.thread = threading.Thread(target=lambda: status.append(desktop.notify(
            "Transcribing…", persistent=True, icon=MIC_ICON, replace=indicator)), daemon=True)
        status.thread.start()
        try:
            with self.turn:
                self.turn.wait_for(lambda: self.serving == ticket)
            self.process(chunks, excess, rewrite, warming, status)
        except Exception:  # noqa: BLE001 - a failed take must not pass silently
            traceback.print_exc()
            end_status(status)
            desktop.notify("Dictation failed", level="error",
                           body="Try again. If it keeps failing, see: journalctl --user -u tuxwhisper")
        finally:
            end_status(status)
            with self.turn:
                self.serving += 1
                self.turn.notify_all()

    def process(self, chunks, excess, rewrite, warming, status=None):
        """Transcribe one take, clean it up if asked, and paste it. status holds the take's
        "Transcribing…" notification, closed once there's a result or a notice to show."""
        audio = self.np.concatenate(chunks)[:, 0] if chunks else []
        if excess:
            audio = audio[:-excess]
        if len(audio) < store.RATE * 0.3:  # too short to be speech; Whisper hallucinates on these
            return
        # Read the clipboard to restore after the paste while the take transcribes.
        snapshot = self.snapshot_clipboard() if self.kbd is not None else None
        if not self.models.ensure_loaded():
            return
        t0 = time.time()
        result = self.models.transcribe(audio, store.settings())
        if result is None:
            return
        text, lang = result
        self.last_used = time.monotonic()
        print(f"[{len(audio) / store.RATE:.1f}s audio (-{excess / store.RATE:.2f}s tail), {time.time() - t0:.2f}s asr] {text!r}",
              flush=True)
        missed = missed_speech(len(audio) / store.RATE, text)
        if missed:
            print(f"  {missed[0]}: {missed[1]}", flush=True)
            end_status(status)
            desktop.notify(missed[0], body=missed[1], level="warn")
        if not text:
            return
        # Saved takes keep Whisper's text, matching the audio.
        with self.lock:
            self.takes += 1
            take = self.last_take = Take(audio, text, id=self.takes)
        if rewrite:
            t0 = time.time()
            # Say so only when the cleanup is slow (the model is still loading, usually), in place
            # of "Transcribing…": the same notification, so nothing flashes up and away.
            status = status if status is not None else []
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
            print(f"  [{time.time() - t0:.2f}s rewrite] {text!r}", flush=True)
        text = store.apply_replacements(text)
        end_status(status)
        typed = self.paste(text + " ", snapshot)
        take.entry = store.add_history(text, rewritten=rewrite, typed=typed)  # only a typed take can be retyped
        self.warn_if_clipped(audio)
        if time.monotonic() - self.last_hint > HINT_EVERY:  # in case the key changed; not every take
            self.last_hint = time.monotonic()
            threading.Thread(target=self.refresh_hint, daemon=True).start()

    def refresh_hint(self):
        """The settings key, shown under "Recording…"; looked up off the hot path (about 50 ms)."""
        self.last_hint = time.monotonic()
        key = desktop.settings_shortcut()
        self.hint = f"<i>⚙ {key}</i>" if key else None  # GNOME allows only b/i/u markup

    def warn_if_clipped(self, audio):
        """Tell the user, after the paste, when the mic was too loud; at most every 10 minutes."""
        clipped = float((self.np.abs(audio) >= CLIP_LEVEL).mean())
        if clipped > CLIP_SHARE and time.monotonic() - self.last_clip_warning > CLIP_WARN_EVERY:
            self.last_clip_warning = time.monotonic()
            print(f"  clipping: {clipped:.2%} of samples", flush=True)
            desktop.notify("Your mic is clipping", level="warn",
                           body="Move back a little, or lower its input volume in Settings → Sound.")

    def save(self):
        """Offer to save the last take for fine-tuning, after the user checks its transcript."""
        with self.lock:  # each take is saved at most once
            take, self.last_take = self.last_take, None
        if take is None:
            if time.monotonic() - self.last_save < Dictation.REPEAT_WINDOW:
                return  # auto-repeat of the key that just saved
            desktop.notify("Nothing to save")
            return
        self.last_save = time.monotonic()
        # Ask first, off the socket loop: dictation keys keep working while the dialog is open.
        threading.Thread(target=self.confirm_save, args=(take,), daemon=True).start()

    def confirm_save(self, take):
        """Show the transcript for checking (a saved one must match the audio exactly), then
        save. Cancelled or failed: nothing is saved, and the take can still be saved with the key."""
        try:
            confirmed = desktop.edit_text(take.text, "Save for fine-tuning", "Save",
                                          "Check the transcript matches what you said, then Save. "
                                          "Cancel keeps it; you can save it later.")
        except Exception:  # noqa: BLE001 - e.g. no dialog tool: say so rather than seem to ignore the key
            traceback.print_exc()
            desktop.notify("Couldn't open the save dialog", body="See the log: journalctl --user -u tuxwhisper",
                           level="error")
            self.restore_take(take)
            return
        if confirmed is None or not confirmed.strip():
            self.restore_take(take)
            return
        try:
            self.write_recording(take.audio, confirmed.strip(), take.entry)
        except Exception as e:  # noqa: BLE001 - e.g. a full disk: keep the take for another try
            traceback.print_exc()
            desktop.notify("Couldn't save the recording", body=write_error(e), level="error")
            self.restore_take(take)

    def restore_take(self, take):
        """Let the save key offer take again, unless a newer take came while its dialog was open."""
        with self.lock:
            if self.last_take is None and self.takes == take.id:
                self.last_take = take

    def write_recording(self, audio, text, entry):
        """Write the take as recordings/<timestamp>.wav + a row in metadata.csv (Hugging Face
        `audiofolder` layout), and note it on its history entry for Correct last transcript."""
        os.makedirs(store.RECORDINGS, mode=0o700, exist_ok=True)
        os.chmod(store.RECORDINGS, 0o700)  # voice data: only you can read it
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav"
        with wave.open(os.path.join(store.RECORDINGS, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(store.RATE)
            w.writeframes((self.np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        meta = os.path.join(store.RECORDINGS, "metadata.csv")
        with store.file_lock(meta):
            new = not os.path.exists(meta)
            takes, frames = (0, 0) if new else store.dataset_frames(meta)  # before this take's row
            with open(meta, "a", newline="") as f:
                out = csv.writer(f)
                if new:
                    out.writerow(["file_name", "transcription"])
                out.writerow([name, text])
            takes, frames = takes + 1, frames + len(audio)
            store.save_dataset_size(meta, takes, frames)
        seconds = frames / store.RATE
        total = f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"
        if entry:
            store.update_history_entry(entry, recording=name)
        else:
            store.mark_last_saved(name)
        saved = f"{desktop.plural(takes, 'recording', 'recordings')}, {total}"
        print(f"saved {name} ({saved})", flush=True)
        desktop.notify(f"Saved ({saved})")

    def retype(self):
        """Replace the last dictation with its correction (from retype_path()): erase what was
        typed with BackSpace, then paste the fix. False if there's no virtual keyboard."""
        try:
            with open(store.retype_path()) as f:
                job = json.load(f)
            os.remove(store.retype_path())
        except (OSError, ValueError, TypeError):
            return False
        with self.lock:
            # Mid-take, the text before the cursor may not be the one to correct.
            if self.kbd is None or self.stream is not None or self.next_ticket != self.serving:
                return False
            # Take a turn like a take: one stopped meanwhile pastes after this, not amid it.
            self.next_ticket += 1
        try:
            time.sleep(0.4)  # the correction dialog just closed; let focus return to the app
            k, ec = self.kbd, self.ec
            for _ in range(job["delete"]):
                for val in (1, 0):
                    k.write(ec.EV_KEY, ec.KEY_BACKSPACE, val)
                    k.syn()
                time.sleep(0.004)
            return self.paste(job["text"] + " ")
        finally:
            with self.turn:
                self.serving += 1
                self.turn.notify_all()

    def snapshot_clipboard(self):
        """Start reading the selections the paste key overwrites; paste() takes the result."""
        selections = paste_selections(store.settings().paste_key)
        saved = {}
        thread = threading.Thread(target=lambda: saved.update(read_selections(selections)), daemon=True)
        thread.start()
        return thread, saved

    def paste(self, text, snapshot=None):
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
        ec, k = self.ec, self.kbd
        presses = [(c, 1) for c in keys] + [(c, 0) for c in reversed(keys)]
        for code, val in presses:
            k.write(ec.EV_KEY, code, val)
            k.syn()
            time.sleep(0.01)
        time.sleep(0.3)  # the app fetches the clipboard asynchronously after the paste
        write_selections({sel: old for sel, old in saved.items() if old is not None})
        return True


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



def write_error(e):
    """A failed save in plain words: the common causes said plainly, anything else as is."""
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return "The disk is full."
    if isinstance(e, OSError) and e.errno in (errno.EACCES, errno.EPERM):
        return f"No permission to write to {store.RECORDINGS}."
    return str(e)


def missed_speech(seconds, text):
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


def repeat_window():
    """How soon a second press counts as the key's auto-repeat: the first repeat comes after
    the desktop's repeat delay (GNOME's default 500 ms), so allow that plus a margin. Users
    with a longer delay (common for accessibility) would otherwise have a hold cut short."""
    try:
        r = subprocess.run(["gsettings", "get", "org.gnome.desktop.peripherals.keyboard", "delay"],
                           capture_output=True, text=True, timeout=2)
        return int(r.stdout.split()[-1]) / 1000 + 0.2
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return Dictation.REPEAT_WINDOW



def daemon_running(sock):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(sock)
        return True
    except OSError:
        return False


def serve():
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


def handle(d, conn):
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


def typed_length(text):
    """How many BackSpaces erase text: one per character, not per combining accent."""
    return sum(1 for c in text if not unicodedata.combining(c))
