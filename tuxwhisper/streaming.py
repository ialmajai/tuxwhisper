"""Long takes transcribed in parts while they're still being recorded, so the text comes soon
after the key's release. As in whisper_streaming (LocalAgreement): each pass re-transcribes the
audio after the words agreed so far; the words two passes in a row agree on are kept, and the
rest stay provisional, so a later pass, hearing more of the sentence, can still correct them."""
import logging
import re
import threading
import time

from tuxwhisper import store

log = logging.getLogger(__name__)

STREAM_AFTER = 20  # seconds recorded before the first pass: shorter takes are transcribed whole
PASS_EVERY = 2     # seconds of new audio between passes (and never two at once)
# Seconds after the agreed words with no agreement: keep words anyway, as a pass must fit in
# Whisper's 30 s window. Late enough that a slow first pass (5 s on a CPU) still gets a second
# one to agree with.
FORCE_AFTER = 27
MAX_WINDOW = 30
PAUSE = 0.4        # seconds between words that count as a pause: a safe place to cut
# Seconds each pass starts before the agreed words end. Word timestamps can be late, and a pass
# starting at the last word's end would clip the next word ("wouldn't" heard as "ouldn't");
# words heard again in the overlap are dropped (see repeated()).
OVERLAP = 0.5
SLACK = 0.2  # seconds past the cut a word heard again in the overlap may be said to start
CONTEXT_CHARS = 200  # the end of the text so far, given as the prompt for the next pass
TICK = 0.25        # seconds between checks for enough new audio


def normal(word: str) -> str:
    """A word as compared between passes: lower case, no punctuation."""
    return re.sub(r"\W", "", word.lower())


def agreement(previous: list, current: list) -> int:
    """How many words two passes over the same audio agree on, from the start."""
    n = 0
    for (a, _, _), (b, _, _) in zip(previous, current, strict=False):
        if normal(a) != normal(b):
            break
        n += 1
    return n


def pause_after(words: list, i: int) -> bool:
    """Whether word i is followed by a pause before the next word."""
    return i + 1 < len(words) and words[i + 1][1] - words[i][2] >= PAUSE * store.RATE


def repeated(kept: list, words: list, cut: int) -> int:
    """How many of a pass's first words were heard again in the overlap: they repeat the last
    kept ones and start before the cut (give or take; timestamps there are rough). A phrase
    the speaker really repeated runs on past the cut."""
    for n in range(min(5, len(kept), len(words)), 0, -1):
        if (words[n - 1][1] < cut + SLACK * store.RATE
                and [normal(w) for w, _, _ in kept[-n:]] == [normal(w) for w, _, _ in words[:n]]):
            return n
    return 0


def joined(words: list) -> str:
    """Words as Whisper spells them, each with its own leading space (none in Chinese or Japanese)."""
    return "".join(w for w, _, _ in words).strip()


class LongTake:
    """One take's passes while it's recorded: the agreed words, where in the audio they end,
    the take's language, and whether streaming failed (the take is then transcribed whole,
    as a short one is). Words are (text, start, end), in samples from the take's start."""

    def __init__(self, models, chunks: list, settings):
        """models: the engine.ModelManager; chunks: the take's list of mic blocks, which the
        mic's callback keeps adding to; settings: the take's store.Settings."""
        self.models, self.chunks, self.settings = models, chunks, settings
        self.words = []       # agreed: kept for the final text
        self.committed = 0    # where the agreed words end: the next pass starts just before
        self.hypothesis = []  # the last pass's words after the agreed ones, still provisional
        self.lang = settings.language or None  # else the first pass that hears words decides it
        self.failed = False
        self.passes = 0
        self.passed_at = 0  # samples recorded at the last pass
        self.tail = 0.0  # seconds of audio in the last pass, after the key's release
        self.stopped = threading.Event()
        self.thread = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._work, daemon=True, name=f"{store.APP}-stream")
        self.thread.start()

    def _work(self):
        while not self.stopped.wait(TICK) and self.tick():
            pass

    def stop(self) -> None:
        """The recording stopped: start no more passes (one under way finishes)."""
        self.stopped.set()

    def tick(self) -> bool:
        """Run a pass if there's enough new audio; False once streaming has failed."""
        chunks = list(self.chunks)  # the mic's callback adds to it meanwhile
        recorded = sum(len(c) for c in chunks)
        if recorded < STREAM_AFTER * store.RATE or recorded - self.passed_at < PASS_EVERY * store.RATE:
            return True
        import numpy as np
        self.passed_at = recorded
        try:
            self.step(np.concatenate(chunks)[:, 0])
        except Exception:  # the take is still transcribed whole at the stop
            log.exception("  transcribing while recording failed; the take will be transcribed whole")
            self.failed = True
        return not self.failed

    def step(self, audio) -> None:
        """One pass over the audio after the agreed words; audio is the take so far."""
        rate = store.RATE
        if len(audio) - self.committed > FORCE_AFTER * rate:
            self.force_commit()
        if not self.failed and len(audio) - self.pass_start() > MAX_WINDOW * rate:
            log.info("  no place to cut within Whisper's window; the take will be transcribed whole")
            self.failed = True
        if self.failed:
            return
        t0 = time.time()
        window = (len(audio) - self.pass_start()) / rate
        # Not decoded again when the text looks wrong: a short window sometimes comes out as one
        # phrase repeated, and its retries took 25 s instead of 2 on the CPU, which a release
        # would wait for. A wrong pass just doesn't agree with the next one.
        words = self.transcribe(audio, fallback=False)
        if words is None:
            self.failed = True
            return
        agreed = agreement(self.hypothesis, words)
        self.commit(words, agreed)
        self.passes += 1
        log.info(f"  pass {self.passes}: {window:.1f} s in {time.time() - t0:.2f}s → {len(words)} words, "
                 f"{agreed} agreed; {len(self.words)} kept, to {self.committed / rate:.1f} s")

    def pass_start(self) -> int:
        """Where the next pass starts: a little before the agreed words end (OVERLAP)."""
        return max(0, self.committed - int(OVERLAP * store.RATE))

    def transcribe(self, audio, fallback=True) -> list | None:
        """The words after the agreed ones in audio, the take so far; None if the model
        couldn't run."""
        base = self.pass_start()
        result = self.models.transcribe_part(audio[base:], self.settings, self.lang, self.context(), fallback)
        if result is None:
            return None
        words, lang = result
        if words and self.lang is None:
            self.lang = lang  # every later pass keeps it: a part may be too short to tell
        rate = store.RATE
        words = [(w, base + int(start * rate), base + int(end * rate)) for w, start, end in words]
        return words[repeated(self.words, words, self.committed):]

    def context(self) -> str:
        """The end of the agreed text, from a word's start: Whisper continues its style."""
        text = joined(self.words)
        if len(text) <= CONTEXT_CHARS:
            return text
        return text[-CONTEXT_CHARS:].split(" ", 1)[-1]

    def commit(self, words: list, agreed: int) -> None:
        """Keep the first agreed words, up to the last one followed by a pause if there is one:
        a cut in a pause can't split a word whose timestamps are a little off."""
        self.keep(words, next((i + 1 for i in reversed(range(agreed)) if pause_after(words, i)), agreed))

    def force_commit(self) -> None:
        """No agreement for too long (e.g. Whisper keeps changing its mind): keep the last pass's
        words up to its last pause, else all but its last word, which the recording may have cut
        off. With no words to keep, give up: the take is transcribed whole."""
        words = self.hypothesis
        ends = range(len(words) - 1)
        k = next((i + 1 for i in reversed(ends) if pause_after(words, i)), len(words) - 1)
        if k <= 0:
            log.info(f"  no words agreed in {FORCE_AFTER} s; the take will be transcribed whole")
            self.failed = True
            return
        log.info(f"  no agreement in {FORCE_AFTER} s; keeping {k} words")
        self.keep(words, k)

    def keep(self, words: list, k: int) -> None:
        """Add words[:k] to the agreed ones and start the next pass after them: halfway through
        the pause after them if there is one, else at the last one's end."""
        if k:
            self.words += words[:k]
            last = words[k - 1][2]
            self.committed = (last + words[k][1]) // 2 if pause_after(words, k - 1) else last
        self.hypothesis = words[k:]

    def finish(self, audio) -> str | None:
        """The take's text once it stopped: the agreed words, then a last pass over the rest
        (provisional words included), which ends at the key's release (audio is the take with
        the excess dropped). Waits for a pass under way. None if nothing was agreed or streaming
        failed: the take is then transcribed whole."""
        self.stop()
        if self.thread is not None:
            self.thread.join()
        if self.failed or not joined(self.words):
            return None
        self.tail = (len(audio) - self.committed) / store.RATE
        rest = ""
        if self.tail >= 0.3:  # shorter is no speech, and Whisper invents some
            words = self.transcribe(audio)
            if words is None:
                return None
            rest = joined(words)
        return f"{joined(self.words)} {rest}".strip()
