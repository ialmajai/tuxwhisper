"""Tests for the speech models: whispered words, GPU memory and language detection."""
import io
import types
import urllib.error

import numpy as np
import pytest
from conftest import LanguageModel, OOMModel, burst, fake_gpu, make_dictation, notified

from tuxwhisper import engine, store


@pytest.mark.parametrize("free, expected, tried", [
    (2.0, "medium", ["medium"]),
    (1.2, "small", ["medium", "small"]),
    (0.3, "base", ["medium", "small", "base"]),
    (0.1, "tiny", ["medium", "small", "base", "tiny"]),
    (0.05, None, ["medium", "small", "base", "tiny"]),
])
def test_load_smaller_tries_each_smaller_model_largest_first(monkeypatch, free, expected, tried):
    fake_gpu(monkeypatch, [free], [])
    names = []
    found = engine.load_smaller("large-v3-turbo", names)
    assert (found[1] if found else None) == expected
    assert names == tried


def test_load_smaller_uses_english_only_models_only_for_english(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [3.0], loads)
    assert engine.load_smaller("large-v3")[1] == "large-v3-turbo"
    store.save_settings({"language": "en"})
    assert engine.load_smaller("large-v3-turbo")[1] == "distil-large-v3.5"


def test_chosen_model_is_loaded_once_it_fits_after_none_did_at_startup(monkeypatch):
    loads, free = [], [0.05]
    fake_gpu(monkeypatch, free, loads)
    models = engine.ModelManager(*engine.load_model())  # nothing fits: no model yet
    assert models.model is None and not models.ensure_loaded()
    free[0] = 3.0  # another program freed the GPU
    monkeypatch.setattr(engine, "free_gpu_gb", lambda: None)  # e.g. Vulkan: no memory figure
    assert models.ensure_loaded()
    assert loads == [engine.asr_model()] and models.name == "large-v3-turbo" and not models.fallback_from


def test_startup_out_of_memory_falls_back(monkeypatch):
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    _, name, replaced = engine.load_model()
    assert loads == ["small"] and name == "small" and replaced == "large-v3-turbo"


def test_startup_with_no_room_waits_instead_of_exiting(monkeypatch):
    fake_gpu(monkeypatch, [0.05], [])
    assert engine.load_model() == (None, None, "large-v3-turbo")
    assert notified == ["Not enough GPU memory to dictate"]


def test_download_verified_refuses_a_wrong_checksum(monkeypatch, tmp_path):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass
    monkeypatch.setattr(engine.urllib.request, "urlopen", lambda url, timeout: Resp(b"model"))
    good = engine.hashlib.sha256(b"model").hexdigest()
    engine.download_verified("http://x", str(tmp_path / "ok.bin"), good)
    assert (tmp_path / "ok.bin").read_bytes() == b"model"
    with pytest.raises(OSError):
        engine.download_verified("http://x", str(tmp_path / "bad.bin"), "0" * 64)
    assert not (tmp_path / "bad.bin").exists()


def test_sound_seconds_counts_sound_not_the_whole_take():
    assert engine.sound_seconds(np.zeros(4 * 16000, dtype=np.float32)) == 0.0
    assert engine.sound_seconds(burst(0.5)) == pytest.approx(0.5, abs=0.06)


def test_sound_seconds_counts_a_take_that_is_sound_throughout():
    assert engine.sound_seconds(burst(4.0)) == pytest.approx(4.0, abs=0.06)


def test_sound_seconds_ignores_a_click_in_digital_silence():
    a = np.zeros(4 * 16000, dtype=np.float32)
    a[30000:30010] = 0.01
    assert engine.sound_seconds(a) <= 0.02  # the click's own frame at most, not the whole take


def test_syllables_count_each_cjk_character():
    assert engine.syllables("你好") == 2
    assert engine.syllables("ありがとう") == 5
    assert engine.syllables("안녕하세요") == 5
    assert engine.syllables("Да") == 1  # other scripts: about one per 2.5 letters


@pytest.mark.parametrize("text, sound, ok", [
    ("Yes.", 0.3, True),
    ("Thank you.", 0.3, True),                 # 6.7 syllables a second: fast but possible
    ("God bless you.", 0.3, False),            # invented from a whispered "yes": 10 a second
    ("This is a sentence, with punctuation and capitals.", 2.0, False),  # the prompt echoed
    (". . . . .", 1.0, False),
])
def test_plausible(text, sound, ok):
    assert engine.plausible(text, burst(sound)) is ok


def test_failed_warm_up_frees_the_model(monkeypatch):
    freed = []

    class Broken:
        def transcribe(self, audio, **_):
            raise RuntimeError("CUDA failed with error out of memory")

        def unload(self, to_cpu=True):
            freed.append(True)

    monkeypatch.setattr(engine, "DEVICE", "vulkan")
    monkeypatch.setattr(engine, "WhisperCpp", lambda name: Broken())
    with pytest.raises(RuntimeError):
        engine.load_whisper("tiny")
    assert freed == [True]


def test_language_is_detected_in_the_same_pass_and_guessed_from_the_last_take(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = LanguageModel("fr")
    assert d.models.transcribe(burst(1), store.settings()) == ("hi", "fr")
    assert d.models.model.calls == [(None, True), ("fr", False)]  # guessed English: redone without its prompt
    d.models.model.calls.clear()
    assert d.models.transcribe(burst(1), store.settings()) == ("hi", "fr")
    assert d.models.model.calls == [(None, False)]  # guessed right: one pass


def test_set_language_is_used_without_detection(monkeypatch):
    d = make_dictation(monkeypatch)
    store.save_settings({"language": "de"})
    d.models.model = LanguageModel("fr")
    assert d.models.transcribe(burst(1), store.settings()) == ("hi", "de") and d.models.model.calls == [("de", False)]


def test_model_manager_steps_down_and_back_on_its_own(monkeypatch):
    free, loads = [1.2], []
    fake_gpu(monkeypatch, free, loads)
    m = engine.ModelManager(OOMModel())
    assert m.ensure_loaded() and m.name == "small" and m.fallback_from == "large-v3-turbo"
    assert m.transcribe(burst(1), store.settings()) == ("take1", "en")
    free[0] = 4.0
    m.restore_if_room()
    assert loads == ["small", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"]
    assert m.fallback_from is None and m.name == "large-v3-turbo"


def test_missing_speech_library_is_reported_not_exited(monkeypatch):
    monkeypatch.setattr(engine, "model_downloaded", lambda name: True)
    def missing(name):
        raise ImportError("No module named 'faster_whisper'", name="faster_whisper")
    monkeypatch.setattr(engine, "load_whisper", missing)
    with pytest.raises(engine.ModelUnavailable, match="faster_whisper is missing"):
        engine.load_model()


@pytest.fixture
def fake_ggml(monkeypatch, tmp_path):
    """pywhispercpp's constants (its library isn't loaded in tests) and a fake model server;
    the server's reply is set as (status, announced length, body)."""
    import io
    import sys
    import urllib.error
    monkeypatch.setitem(sys.modules, "pywhispercpp.constants", types.SimpleNamespace(
        AVAILABLE_MODELS=["base"], MODELS_BASE_URL="https://hf.example", MODELS_DIR=str(tmp_path),
        MODELS_PREFIX_URL="resolve/main/ggml"))
    reply = {"status": 200, "length": 6, "body": b"ggml.."}
    def urlopen(url, timeout=None):
        if reply["status"] != 200:
            raise urllib.error.HTTPError(url, reply["status"], "error", {}, io.BytesIO(b"<html>"))
        r = io.BytesIO(reply["body"])
        r.headers = {"Content-Length": str(reply["length"])}
        return r
    monkeypatch.setattr(engine.urllib.request, "urlopen", urlopen)
    return reply, tmp_path / "ggml-base.bin"


def test_ggml_download_is_saved_only_once_complete(fake_ggml):
    _, path = fake_ggml
    engine.download_ggml("base")
    assert path.read_bytes() == b"ggml.." and not path.with_suffix(".bin.part").exists()


@pytest.mark.parametrize("status, length", [(200, 1000), (404, 6)])
def test_failed_ggml_download_leaves_no_model_file(fake_ggml, status, length):
    reply, path = fake_ggml
    reply.update(status=status, length=length)  # cut off part-way, or an error page
    with pytest.raises(OSError):
        engine.download_ggml("base")
    assert not path.exists() and not path.with_suffix(".bin.part").exists()


def test_ggml_model_already_there_is_not_downloaded_again(fake_ggml):
    reply, path = fake_ggml
    path.write_bytes(b"old")
    reply["status"] = 500  # would fail if asked
    engine.download_ggml("base")
    assert path.read_bytes() == b"old"


@pytest.mark.parametrize("error, notice", [
    (urllib.error.URLError("no network"), "Couldn't download the speech model"),
    (engine.DownloadError("didn't match its checksum"), "Couldn't download the speech model"),
    (OSError("libcublas.so.12: cannot open shared object file"), "Dictation can't use the GPU"),
])
def test_failed_download_at_startup_isn_t_called_a_gpu_problem(monkeypatch, error, notice):
    monkeypatch.setattr(engine, "DEVICE", "vulkan")
    monkeypatch.setattr(engine, "model_downloaded", lambda name: True)
    def load(name):
        raise error
    monkeypatch.setattr(engine, "load_whisper", load)
    with pytest.raises(engine.ModelUnavailable):
        engine.load_model()
    assert notified[-1] == notice
