"""The dictation daemon: the mic, key presses and holds, and taking turns to paste."""
import json
import logging
import os
import threading
import time
import traceback
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor

from tuxwhisper import desktop, engine, llm, store
from tuxwhisper.client import (  # noqa: F401 - re-exported for menu and tests
    request,
    send,
    sock_path,
)
from tuxwhisper.pipeline import (
    MIC_ICON,
    Paster,
    Status,
    TakeContext,
    TakePipeline,
    end_status,
)
from tuxwhisper.recordings import Recordings

log = logging.getLogger(__name__)

HINT_EVERY = 300  # seconds between lookups of the settings key shown under "Recording…"


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
            log.info(f"virtual keyboard unavailable ({e}); transcripts go to the clipboard")
            desktop.notify("No virtual keyboard", body="Transcripts are copied instead. Paste them with Ctrl+V.",
                           level="warn")
        d = cls(models, sd, kbd, ecodes)
        d.refresh_hint()
        d.REPEAT_WINDOW = desktop.repeat_window(Dictation.REPEAT_WINDOW)
        threading.Thread(target=d._watch_release, daemon=True).start()
        threading.Thread(target=d._watch_idle, daemon=True).start()
        log.info("ready")
        return d

    def __init__(self, models, sd, kbd, ec):
        """Dictation's state around the speech model (an engine.ModelManager), the sounddevice
        module, the virtual keyboard (None: copy instead of typing) and evdev's ecodes. create()
        passes the real ones and starts the watcher threads; tests pass fakes."""
        self.sd = sd
        self.models = models
        self.paster = Paster(kbd, ec)
        self.recordings = Recordings()
        self.pipeline = TakePipeline(models, self.paster, self.recordings, after_paste=self.maybe_refresh_hint)
        self.lock = threading.Lock()
        self.press_lock = threading.Lock()  # presses come from one thread per connection
        # Takes are transcribed and pasted one at a time, in the order they stopped.
        self.turn = threading.Condition()
        self.next_ticket = self.serving = 0
        self.stream = None
        self.indicator = None  # ID of the on-screen recording notification
        self.hint = None  # the settings key shown under "Recording…"; see refresh_hint()
        self.last_hint = -HINT_EVERY
        self.chunks = []
        self.dropped = 0  # lost-audio count for the take being recorded
        self.last_press = 0.0
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
        self.models.unload_if_idle(store.settings().asr_unload_minutes, idle=self.stream is None)

    def _start(self, rewrite):
        """Open the mic for a new take (under self.lock); the stream, or None if the mic failed."""
        self.models.touch()
        threading.Thread(target=self.models.ensure_loaded, daemon=True).start()  # overlaps the speech
        self.chunks = []
        self.dropped = 0
        self.rewrite = rewrite  # mode is set by the key that started the take
        # A cold load takes seconds, so overlap it with the speech; the cleanup skips its own
        # GPU check when the warm-up confirmed it.
        self.warming = self.background.submit(llm.warm_rewrite_model) if rewrite else None
        desktop.sound("device-added")  # 0.2 s; over before the mic finishes opening
        try:
            stream = self.sd.InputStream(
                samplerate=store.RATE, channels=1, dtype="float32",
                callback=self._capture)
            stream.start()
        except Exception as e:  # noqa: BLE001 - unplugged, busy or no mic: stay ready
            log.info(f"microphone unavailable: {e}")
            desktop.notify("Microphone unavailable", body="Check that it's connected and not in use.",
                           level="error")
            return None
        self.stream, self.indicator = stream, None
        return stream

    def _capture(self, data, frames=None, time_info=None, status=None):
        """The mic's callback: keep each block, and count lost audio, which would otherwise only
        show as a worse transcript."""
        if status and status.input_overflow:
            self.dropped += 1
        self.chunks.append(data.copy())

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

    def _stop(self, cut_at) -> TakeContext:
        """Close the mic (under self.lock); the take, with what it needs captured now."""
        # Drop the audio recorded between the key release (cut_at) and now.
        excess = int(max(0.0, time.monotonic() - cut_at) * store.RATE) if cut_at else 0
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as e:  # noqa: BLE001 - e.g. the mic was unplugged mid-take
            log.info(f"microphone error on stop: {e}")
        finally:
            self.stream = None
        indicator, self.indicator = self.indicator, None
        desktop.sound("device-removed")
        chunks, self.chunks = self.chunks, []
        # This take's own mode: the next take may start before this one is pasted.
        ticket, self.next_ticket = self.next_ticket, self.next_ticket + 1
        return TakeContext(chunks, excess, self.rewrite, self.warming, ticket, indicator,
                           dropped=self.dropped)

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
        # Swap "Recording…" for "Transcribing…" in place: one notification, not a new popup.
        # Shown in the background: notify-send -p takes a while, and the take needn't wait.
        status = take.status = Status()
        status.thread = threading.Thread(target=lambda: status.append(desktop.notify(
            "Transcribing…", persistent=True, icon=MIC_ICON, replace=take.indicator)), daemon=True)
        status.thread.start()
        try:
            with self.turn:
                self.turn.wait_for(lambda: self.serving == take.ticket)
            self.pipeline.run(take)
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

    def maybe_refresh_hint(self):
        """After a take: look the settings key up again in case it changed, but not every take."""
        if time.monotonic() - self.last_hint > HINT_EVERY:
            self.last_hint = time.monotonic()
            threading.Thread(target=self.refresh_hint, daemon=True).start()

    def refresh_hint(self):
        """The settings key, shown under "Recording…"; looked up off the hot path (about 50 ms)."""
        self.last_hint = time.monotonic()
        key = desktop.settings_shortcut()
        self.hint = f"<i>⚙ {key}</i>" if key else None  # GNOME allows only b/i/u markup

    def save(self):
        """Offer to save the last take for fine-tuning, after the user checks its transcript."""
        self.recordings.save(Dictation.REPEAT_WINDOW)

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
            if self.paster.kbd is None or self.stream is not None or self.next_ticket != self.serving:
                return False
            # Take a turn like a take: one stopped meanwhile pastes after this, not amid it.
            self.next_ticket += 1
        try:
            time.sleep(0.4)  # the correction dialog just closed; let focus return to the app
            self.paster.erase(job["delete"])
            return self.paster.paste(job["text"] + " ")
        finally:
            with self.turn:
                self.serving += 1
                self.turn.notify_all()


def typed_length(text):
    """How many BackSpaces erase text: one per character, not per combining accent."""
    return sum(1 for c in text if not unicodedata.combining(c))
