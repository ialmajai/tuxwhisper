"""Saving takes for fine-tuning: the save key's last take, its check dialog and the files."""
import csv
import datetime
import errno
import logging
import os
import threading
import time
import traceback
import wave
from dataclasses import dataclass
from typing import Any

from tuxwhisper import desktop, store

log = logging.getLogger(__name__)


@dataclass
class Take:
    """A transcribed take, kept for the save key: its audio, Whisper's text (matching the
    audio), its history entry once pasted, and its number among the daemon's takes."""
    audio: Any
    text: str
    entry: dict | None = None
    id: int = 0


class Recordings:
    """The last take, until the save key takes it, and writing it to the recordings folder."""

    def __init__(self):
        self.lock = threading.Lock()
        self.last_take: Take | None = None
        self.takes = 0  # counts takes, so a cancelled save can tell whether a newer one came
        self.last_save = 0.0

    def keep(self, audio: Any, text: str) -> Take:
        """Make a new take the one the save key offers."""
        with self.lock:
            self.takes += 1
            take = self.last_take = Take(audio, text, id=self.takes)
        return take

    def save(self, repeat_window: float) -> None:
        """Offer to save the last take for fine-tuning, after the user checks its transcript.
        A press within repeat_window of the last save is the key's auto-repeat."""
        with self.lock:  # each take is saved at most once
            take, self.last_take = self.last_take, None
        if take is None:
            if time.monotonic() - self.last_save < repeat_window:
                return  # auto-repeat of the key that just saved
            desktop.notify("Nothing to save")
            return
        self.last_save = time.monotonic()
        # Ask first, off the socket loop: dictation keys keep working while the dialog is open.
        threading.Thread(target=self.confirm_save, args=(take,), daemon=True).start()

    def confirm_save(self, take: Take) -> None:
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

    def restore_take(self, take: Take) -> None:
        """Let the save key offer take again, unless a newer take came while its dialog was open."""
        with self.lock:
            if self.last_take is None and self.takes == take.id:
                self.last_take = take

    def write_recording(self, audio: Any, text: str, entry: dict | None) -> None:
        """Write the take as recordings/<timestamp>.wav + a row in metadata.csv (Hugging Face
        `audiofolder` layout), and note it on its history entry for Correct last transcript."""
        import numpy as np
        os.makedirs(store.RECORDINGS, mode=0o700, exist_ok=True)
        os.chmod(store.RECORDINGS, 0o700)  # voice data: only you can read it
        name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".wav"
        with wave.open(os.path.join(store.RECORDINGS, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(store.RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
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
        seconds = frames / store.RATE
        total = f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds:.0f} s"
        if entry:
            store.update_history_entry(entry, recording=name)
        else:
            store.mark_last_saved(name)
        saved = f"{desktop.plural(takes, 'recording', 'recordings')}, {total}"
        log.info(f"saved {name} ({saved})")
        desktop.notify(f"Saved ({saved})")


def write_error(e: BaseException) -> str:
    """A failed save in plain words: the common causes said plainly, anything else as is."""
    if isinstance(e, OSError) and e.errno == errno.ENOSPC:
        return "The disk is full."
    if isinstance(e, OSError) and e.errno in (errno.EACCES, errno.EPERM):
        return f"No permission to write to {store.RECORDINGS}."
    return str(e)
