"""Speech models: faster-whisper or whisper.cpp, loading them and fitting them in GPU memory."""
import contextlib
import hashlib
import os
import re
import subprocess
import sys
import threading
import time
import types
import urllib.request
from collections.abc import Iterable
from typing import Any, Protocol

from tuxwhisper import desktop, store

DEVICE = os.environ.get("ASR_DEVICE", "cuda")  # "cuda", "vulkan" (AMD/Intel GPUs) or "cpu"
GPU_MODEL = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
CPU_MODEL = "small"  # large models are too slow for live dictation on most CPUs
VULKAN_MODEL = "large-v3-turbo"  # whisper.cpp's name for the same model
# Speech models offered in the settings menu: name, what it's good for, download size. Both
# engines download them by these names; the distil model exists only for faster-whisper.
ASR_MODELS = [
    ("large-v3-turbo", "Best balance of speed and accuracy (GPU default)", "1.6 GB"),
    ("large-v3", "Most accurate, about twice as slow; needs about 4 GB of GPU memory", "3.1 GB"),
    ("distil-large-v3.5", "English only: fast and accurate", "1.5 GB"),
    ("medium", "Good accuracy, lighter", "1.5 GB"),
    ("small", "Fast; the CPU default", "0.5 GB"),
    ("base", "Very fast, less accurate", "0.15 GB"),
    ("tiny", "Fastest, least accurate", "0.08 GB"),
]
# Where faster-whisper downloads each menu model from, to check the cache without importing it.
ASR_REPOS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    **{n: f"Systran/faster-whisper-{n}" for n in ("large-v3", "medium", "small", "base", "tiny")},
}
# Silero voice activity detection for whisper.cpp; faster-whisper ships its own copy.
VAD_MODEL_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
VAD_MODEL_SHA256 = "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf"
# Whisper imitates the style of its prompt. large-v3-turbo often drops capitals and
# punctuation on casual speech; a punctuated prompt brings them back. Set to "" to disable.
PROMPT = os.environ.get("ASR_PROMPT", "Hello. This is a sentence, with punctuation and capitals.") or None


# Whispered words: the voice-activity filter drops them (it looks for voiced speech), while
# Whisper itself hears them fine. Without the filter, though, Whisper sometimes invents a stock
# phrase ("God bless you" from a whispered "yes") or repeats its prompt; those are caught by
# checking the transcript could have been said in the sound there is.
MIN_SOUND = 0.2       # seconds of sound before a take counts as more than silence
MAX_SYLLABLE_RATE = 8  # syllables per second of sound; fast speech is about 6-7
# Frame levels (RMS) bounding the sound threshold. The floor keeps digital silence with a click
# from counting as sound; it sits below a quiet mic's own hiss (about 6e-5 here, three times that
# is the usual threshold), since whispers peak at only 2e-3. The cap keeps a take that is sound
# throughout, whose quietest frames are sound too, from counting as none.
SOUND_FLOOR, SOUND_CAP = 1e-4, 1e-3


def sound_seconds(audio: Any, rate: int = 16000) -> float:
    """How long the take contains sound: 20 ms frames well above its own quietest ones (within
    SOUND_FLOOR and SOUND_CAP). Not the first-to-last span, which breaths and clicks stretch."""
    import numpy as np
    n = rate // 50
    frames = np.asarray(audio[: len(audio) // n * n], dtype="float32").reshape(-1, n)
    if not len(frames):
        return 0.0
    level = np.sqrt((frames ** 2).mean(axis=1)) + 1e-6
    threshold = max(SOUND_FLOOR, min(np.percentile(level, 10) * 3, SOUND_CAP))
    return float((level > threshold).sum()) * n / rate


# Han, Hiragana, Katakana and Hangul syllables.
CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")


def syllables(text: str) -> int:
    """Rough syllable count: vowel groups in Latin-script words, one per character in Chinese,
    Japanese and Korean (each is a syllable), about one per 2.5 letters in other scripts.
    Enough to tell "Yes" from "God bless you"."""
    count = 0
    for word in re.findall(r"\w+", text.lower()):
        cjk = len(CJK.findall(word))
        if cjk:
            count += cjk
            word = CJK.sub("", word)
            if not word:
                continue
        if word.isascii():
            count += max(1, len(re.findall(r"[aeiouy]+", word.removesuffix("e") or word)))
        else:
            count += max(1, round(len(word) / 2.5))
    return count


def prompt_echo(text: str) -> bool:
    """Whether text only repeats (part of) the style prompt, which Whisper does on near-silence."""
    words = re.sub(r"\W+", " ", text.lower()).strip()
    return bool(PROMPT) and bool(words) and words in re.sub(r"\W+", " ", PROMPT.lower())


def plausible(text: str, audio: Any, rate: int = 16000) -> bool:
    """Whether text could have been said in the sound the take contains."""
    if not re.search(r"\w", text) or prompt_echo(text):
        return False
    return syllables(text) / max(sound_seconds(audio, rate), 0.1) <= MAX_SYLLABLE_RATE


class SpeechModel(Protocol):
    """What dictation needs from a speech engine. Segments have .text; info has .language."""

    def transcribe(self, audio: Any, *, language: str | None = None, prompt: str | None = None,
                   hotwords: str | None = None, vad_filter: bool = True) -> tuple[Iterable[Any], Any]: ...

    def detect_language(self, audio: Any) -> tuple[str, float]: ...

    def loaded(self) -> bool: ...

    def load(self) -> None: ...

    def unload(self, to_cpu: bool = True) -> None:
        """Free its GPU memory; to_cpu parks it in RAM for a quick reload, where the engine can."""


class FasterWhisperModel:
    """faster-whisper (CTranslate2) on an NVIDIA GPU or the CPU."""

    def __init__(self, name: str):
        from faster_whisper import WhisperModel
        on_cpu = DEVICE == "cpu"
        self.model = WhisperModel(name, device="cpu" if on_cpu else "cuda",
                                  compute_type="int8" if on_cpu else "float16")

    def transcribe(self, audio, *, language=None, prompt=None, hotwords=None, vad_filter=True):
        return self.model.transcribe(audio, language=language, beam_size=1, vad_filter=vad_filter,
                                     initial_prompt=prompt, hotwords=hotwords)

    def detect_language(self, audio):
        return self.model.detect_language(audio, vad_filter=True)

    def loaded(self):
        return self.model.model.model_is_loaded

    def load(self):
        self.model.model.load_model()

    def unload(self, to_cpu=True):
        self.model.model.unload_model(to_cpu=to_cpu)


class WhisperCpp:
    """whisper.cpp on a Vulkan GPU (AMD, Intel or NVIDIA)."""

    def __init__(self, name: str):
        from pywhispercpp.constants import MODELS_DIR
        from pywhispercpp.utils import redirect_stderr
        self.name = name
        self.quiet = lambda: redirect_stderr(to=None)  # whisper.cpp logs every call to stderr
        self.load()
        # Without VAD, silence comes out as "." or invented phrases, which would be pasted.
        self.vad_model = os.path.join(MODELS_DIR, os.path.basename(VAD_MODEL_URL))
        if not os.path.exists(self.vad_model):
            download_verified(VAD_MODEL_URL, self.vad_model, VAD_MODEL_SHA256)

    def load(self):
        from pywhispercpp.model import Model
        self.model = Model(self.name, print_progress=False, print_realtime=False,
                           redirect_whispercpp_logs_to=None)

    def loaded(self):
        return self.model is not None

    def unload(self, to_cpu=True):  # whisper.cpp can't park a model in RAM, so it reloads from disk later
        self.model = None

    def detect_language(self, audio):
        with self.quiet():
            return self.model.auto_detect_language(audio)[0]

    def transcribe(self, audio, *, language=None, prompt=None, hotwords=None, vad_filter=True):
        # whisper.cpp always runs its voice filter (vad_filter is accepted for the shared call):
        # without it, silence comes out as "." or invented phrases.
        # whisper.cpp has no hotwords; the vocabulary joins the prompt instead.
        prompt = " ".join(p for p in (prompt, hotwords) if p)
        params = {"vad": True, "vad_model_path": self.vad_model}
        if prompt:
            params["initial_prompt"] = prompt
        with self.quiet():
            # Detected here so the caller gets it like faster-whisper's info.language.
            language = language or self.model.auto_detect_language(audio)[0][0]
            params["language"] = language
            return self.model.transcribe(audio, **params), types.SimpleNamespace(language=language)


def download_verified(url: str, path: str, sha256: str, timeout: float = 60) -> None:
    """Download url to path, refusing a file whose SHA-256 doesn't match: it's loaded as native
    code's input, so a corrupted or swapped file must not be used."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise OSError(f"{url} didn't match its checksum; not using it")
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def default_asr_model() -> str:
    return {"cpu": CPU_MODEL, "vulkan": VULKAN_MODEL}.get(DEVICE, GPU_MODEL)


def asr_model() -> str:
    """The speech model picked in the settings menu, else ASR_MODEL, else the engine's default."""
    return store.settings().asr_model or default_asr_model()


def asr_models() -> list[tuple[str, str, str]]:
    """The menu's models this engine can run."""
    return [m for m in ASR_MODELS if DEVICE != "vulkan" or not m[0].startswith("distil")]


def model_downloaded(name: str) -> bool:
    if DEVICE == "vulkan":
        from pywhispercpp.constants import MODELS_DIR
        return os.path.exists(os.path.join(MODELS_DIR, f"ggml-{name}.bin"))
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(ASR_REPOS.get(name, name), "model.bin"), str)


GPU_MARGIN_GB = 0.4  # room for the model's working memory on top of its weights


def out_of_memory(e: BaseException) -> bool:
    return "out of memory" in str(e).lower()


def short_name(name: str) -> str:
    """The menu name of a model given by repo, e.g. large-v3-turbo for the GPU default."""
    return name.split("faster-whisper-")[-1]


def model_size(name: str) -> float:
    """A menu model's size in GB, or infinity for one the menu doesn't know."""
    return next((float(m[2].split()[0]) for m in ASR_MODELS if m[0] == short_name(name)), float("inf"))


def free_gpu_gb() -> float | None:
    """Free memory on the NVIDIA GPU, or None if nvidia-smi can't tell."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        return int(r.stdout.split()[0]) / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def load_smaller(name: str, tried: list[str] | None = None) -> tuple[SpeechModel, str] | None:
    """Try each downloaded model smaller than name, largest first, and return the first that
    loads on the GPU as (model, its name), or None. Names tried are added to tried.
    English-only models are tried only when English is set."""
    english = store.settings().language == "en"
    for cand, _, _ in sorted(asr_models(), key=lambda m: -model_size(m[0])):
        if (model_size(cand) >= model_size(name) or not model_downloaded(cand)
                or (cand.startswith("distil") and not english)):
            continue
        if tried is not None:
            tried.append(cand)
        try:
            return load_whisper(cand), cand
        except RuntimeError as e:
            if not out_of_memory(e):
                raise
            print(f"{cand} doesn't fit either", flush=True)
    return None


def no_memory_notice(tried: list[str] | None = None) -> None:
    """No speech model fits in GPU memory; tried names the ones that didn't."""
    advice = "Close other GPU programs, or set ASR_DEVICE=cpu."
    desktop.notify("Not enough GPU memory to dictate", level="error",
                   body=f"Tried {', '.join(tried)}. {advice}" if tried else advice)


def fallback_notice(name: str, preferred: str) -> None:
    desktop.notify(f"GPU memory is short: using {name}", body=f"Back to {preferred} once there's room.",
                   level="warn")


def load_whisper(name: str) -> SpeechModel:
    """Load a speech model and run it once. Running needs working memory beyond the weights,
    and on a nearly full GPU that's where it runs out, so the out-of-memory fallbacks must
    see it here, at load time, rather than crash the daemon's start or a take."""
    import numpy as np
    print(f"loading {name} on {DEVICE} ...", flush=True)
    model = WhisperCpp(name) if DEVICE == "vulkan" else FasterWhisperModel(name)
    try:
        model.transcribe(np.zeros(16000, dtype=np.float32), vad_filter=False)  # warm-up
    except Exception:
        # Free its GPU memory before a caller tries a smaller one. `del` isn't enough: the
        # traceback being raised keeps the model alive.
        with contextlib.suppress(Exception):
            model.unload(to_cpu=False)
        raise
    return model


def load_model() -> tuple[SpeechModel | None, str | None, str | None]:
    """Whisper on an NVIDIA GPU (default), on another GPU with ASR_DEVICE=vulkan, or on the CPU
    with ASR_DEVICE=cpu (int8, smaller model). A first use downloads the model; a model that
    won't load falls back to the engine's default instead of failing at every start, and one
    that doesn't fit in GPU memory to a smaller one. Returns (model, its name, the model it
    replaced); the model and its name are None when none fits in GPU memory yet."""
    name, default = asr_model(), default_asr_model()
    for attempt in dict.fromkeys([name, default]):
        downloading, short_of_memory = None, False
        try:
            if not model_downloaded(attempt):
                size = next((m[2] for m in ASR_MODELS if m[0] == short_name(attempt)), None)
                about = f"About {size}. " if size else ""
                downloading = desktop.notify(f"Downloading the speech model {attempt}…", persistent=True,
                                     body=f"<i>{about}Dictation starts when it's ready.</i>")
            model = load_whisper(attempt)
        except ImportError as e:
            sys.exit(f"{e.name or 'a speech library'} is missing ({e}); run `tuxwhisper setup` "
                     "(or ./install.sh in a git clone) again")
        except Exception as e:
            desktop.close_notification(downloading)
            if DEVICE != "cpu" and out_of_memory(e):
                print(f"no GPU memory for {attempt} ({e})", flush=True)
                short_of_memory = True
            elif attempt != default:
                print(f"couldn't load {attempt} ({e}); using {default}", flush=True)
                desktop.notify(f"Couldn't load the speech model {attempt}", level="warn",
                               body=f"Using {default} instead. See the log: journalctl --user -u tuxwhisper")
                continue
            elif DEVICE != "cpu":
                desktop.notify("Dictation can't use the GPU", level="error",
                               body="See the log: journalctl --user -u tuxwhisper, or set ASR_DEVICE=cpu.")
                sys.exit(f"{DEVICE} unavailable ({e}); set ASR_DEVICE=cpu to run on the CPU")
            else:
                raise
        if short_of_memory:  # out of the except block, whose traceback keeps the failed model alive
            tried = [short_name(attempt)]
            if found := load_smaller(attempt, tried):
                fallback_notice(found[1], tried[0])
                return found[0], found[1], tried[0]
            # Exiting would restart the daemon every few seconds; wait for a take instead.
            print("no GPU memory for any speech model; retrying on the next take", flush=True)
            no_memory_notice(tried)
            return None, None, tried[0]
        if downloading:
            desktop.close_notification(downloading)
            desktop.notify(f"Speech model {attempt} is ready")
        return model, short_name(attempt), None


RETRY_MAX_SECONDS = 30  # whispered takes longer than this aren't retried without the voice filter


class ModelManager:
    """The speech model in use and what keeps it usable: loading it back after an idle unload,
    stepping down to a smaller model when the GPU runs out of memory (loading or running), and
    going back to the chosen one once there's room."""

    def __init__(self, model: SpeechModel | None, name: str | None = None, fallback_from: str | None = None):
        """model: loaded and warmed up, or None if none fit in GPU memory yet; name: its name;
        fallback_from: the chosen model it stands in for, if it's a smaller one."""
        self.model, self.fallback_from = model, fallback_from
        # The model in use, which may be a smaller one than chosen: out of memory, step down from it.
        self.name = name or short_name(asr_model())
        self.lock = threading.Lock()  # held while the speech model is used, loaded or freed
        self.last_memory_notice = 0.0
        self.last_lang = "en"  # the last take's language: the guess for the next one's prompt

    @classmethod
    def load(cls) -> "ModelManager":
        return cls(*load_model())

    def ensure_loaded(self) -> bool:
        """Load the speech model back onto the GPU if it was freed (under a second from RAM).
        Without enough GPU memory, use the largest smaller model that fits; False if none does."""
        with self.lock:
            if self.model is None:
                return self._step_down(self.fallback_from)
            if self.model.loaded():
                return True
            t0 = time.time()
            try:
                self.model.load()
            except RuntimeError as e:
                if not out_of_memory(e):
                    raise
                print(f"no GPU memory to reload the speech model ({e})", flush=True)
                return self._step_down(self.fallback_from or self.name, self.name)
            print(f"speech model reloaded in {time.time() - t0:.2f}s", flush=True)
            return True

    def _step_down(self, preferred: str, current: str | None = None) -> bool:
        """Swap in the largest model smaller than current (default: preferred, the chosen one)
        that fits in free GPU memory."""
        tried = [current or preferred]
        found = load_smaller(tried[0], tried)
        if not found:
            if time.monotonic() - self.last_memory_notice > 10:  # once per take, not per thread
                self.last_memory_notice = time.monotonic()
                no_memory_notice(tried)
            return False
        self.model, self.name = found
        self.fallback_from = preferred
        fallback_notice(self.name, preferred)
        return True

    def restore_if_room(self) -> None:
        """Go back to the chosen model once the GPU has room for it again; not while the model
        is in use."""
        if self.fallback_from and self.lock.acquire(blocking=False):
            try:
                free = free_gpu_gb()
                if free is None or free < model_size(self.fallback_from) + GPU_MARGIN_GB:
                    return
                try:
                    self.model = load_whisper(asr_model())
                except RuntimeError as e:
                    if not out_of_memory(e):
                        raise
                    return
                self.name = short_name(asr_model())
                desktop.notify(f"Back to {self.fallback_from}", body="The GPU has room for it again.")
                self.fallback_from = None
            finally:
                self.lock.release()

    def unload_if_idle(self, minutes: int, idle: float) -> None:
        """Free the model's GPU memory when it's been idle (seconds) over minutes; 0 = never."""
        if (minutes and DEVICE != "cpu" and idle > minutes * 60
                and self.lock.acquire(blocking=False)):  # not while one is transcribing
            try:
                if self.model is not None and self.model.loaded():
                    self.model.unload()
                    print(f"speech model unloaded after {minutes} min idle", flush=True)
            finally:
                self.lock.release()

    def transcribe(self, audio: Any, settings: "store.Settings") -> tuple[str, str] | None:
        """The take's (text, language); None if the GPU has no memory for any model (the user
        has been told). Out of memory while running, it steps down to a smaller model once."""
        with self.lock:
            short_of_memory = False
            try:
                return self._transcribe(audio, settings)
            except RuntimeError as e:
                if not out_of_memory(e):
                    raise
                # The model is loaded, but another program took the GPU memory it needs to run.
                print(f"no GPU memory to transcribe ({e})", flush=True)
                short_of_memory = True
            if short_of_memory:  # out of the except block, whose traceback holds the model's memory
                self.model.unload()
                if not self._step_down(self.fallback_from or self.name, self.name):
                    return None  # it said so; "Didn't catch that" on top would mislead
                try:
                    return self._transcribe(audio, settings)
                except RuntimeError as e:
                    if not out_of_memory(e):
                        raise
                    print(f"no GPU memory to transcribe with {self.name} either ({e})", flush=True)
                    no_memory_notice()
                    return None

    def _transcribe(self, audio, settings):
        """(text, language) (under self.lock), with whispered takes retried without the voice
        filter."""
        lang = settings.language or None  # "" is automatic
        # The English prompt and vocabulary make Whisper translate other languages into English.
        # With no language set, guess the prompt from the last take's language and let Whisper
        # detect it in the same pass: one transcription instead of a detection and then one.
        english = (lang or self.last_lang) == "en"
        segments, info = self._run(audio, lang, english, vad_filter=True)
        if not lang:
            lang = getattr(info, "language", None)
            if lang is None:  # a model that doesn't report it: detect, as a separate pass
                lang = self.model.detect_language(audio)[0]
            if (lang == "en") != english:  # guessed the wrong prompt: redo with the right one
                english = lang == "en"
                segments, _ = self._run(audio, lang, english, vad_filter=True)
            self.last_lang = lang
        text = " ".join(s.text.strip() for s in segments).strip()
        if (not text and sound_seconds(audio, store.RATE) >= MIN_SOUND
                and len(audio) <= RETRY_MAX_SECONDS * store.RATE):
            # The filter heard no voice in a take with sound: likely whispered. Try without
            # it, keeping the result only if it fits the sound (Whisper may invent a phrase).
            # Not for long takes: a whispered one is short, and a long retry would cost seconds.
            segments, _ = self._run(audio, lang, english, vad_filter=False)
            retry = " ".join(s.text.strip() for s in segments).strip()
            ok = plausible(retry, audio, store.RATE)
            print(f"  no voice found; without the filter: {retry!r} ({'kept' if ok else 'dropped'})", flush=True)
            text = retry if ok else ""
        return text, lang

    def _run(self, audio, lang, english, vad_filter):
        """(segments, info) from the model; lang None detects the language."""
        return self.model.transcribe(audio, language=lang, vad_filter=vad_filter,
                                     prompt=PROMPT if english else None,
                                     hotwords=store.vocab() if english else None)
