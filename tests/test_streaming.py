"""Tests for long takes transcribed in passes while they're recorded."""
import types

import numpy as np
import pytest
from conftest import FakeModel, make_dictation

from tuxwhisper import engine, pipeline, store, streaming
from tuxwhisper.streaming import LongTake

RATE = store.RATE
STEP = 1e-7  # each sample holds its own index (times STEP), so the fake model knows where a pass is


def ramp(seconds):
    return (np.arange(int(seconds * RATE)) * STEP).astype(np.float32)


def speech(seconds, pause_every=None):
    """A script of words 0.5 s apart, ("w0", start, end)...; every pause_every words, a 0.6 s pause."""
    words, t, i = [], 0.2, 0
    while t + 0.4 <= seconds:
        words.append((f"w{i}", t, t + 0.4))
        i += 1
        t += 1.0 if pause_every and i % pause_every == 0 else 0.5
    return words


class ScriptModel(FakeModel):
    """Hears the script's words that lie wholly within the audio it's given. A word in `early`
    is misheard while it ends within a second of the audio's end, as Whisper does a word cut off
    or without what follows. Calls are noted as (start, end) in samples, with their arguments."""
    word_timestamps = True

    def __init__(self, script, lang="en", early=None, fail_at=None):
        super().__init__()
        self.script, self.lang, self.early, self.fail_at = script, lang, early or {}, fail_at
        self.calls = []

    def transcribe(self, audio, language=None, prompt=None, word_timestamps=False, fallback=True, **_):
        start = round(float(audio[0]) / STEP)
        end = start + len(audio)
        self.calls.append(types.SimpleNamespace(start=start, end=end, language=language, prompt=prompt,
                                                words=word_timestamps, fallback=fallback))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("boom")
        words = []
        for w, s, e in self.script:
            s, e = int(s * RATE), int(e * RATE)
            if s >= start and e <= end:
                text = self.early.get(w, w) if end - e < RATE else w
                words.append(types.SimpleNamespace(word=f" {text}", start=(s - start) / RATE,
                                                   end=(e - start) / RATE))
        return ([types.SimpleNamespace(text="".join(w.word for w in words), words=words)],
                types.SimpleNamespace(language=language or self.lang))


def record(long, audio, block=0.5):
    """Feed audio to the take as the mic would, a pass at a time where one is due."""
    n = int(block * RATE)
    for i in range(0, len(audio), n):
        long.chunks.append(audio[i:i + n, None])
        long.tick()


def long_take(monkeypatch, model, seconds, excess=0):
    """A dictation with model, and a take of seconds recorded with passes; (dictation, take)."""
    d = make_dictation(monkeypatch)
    d.models.model = model
    long = LongTake(d.models, [], store.settings())
    record(long, ramp(seconds))
    return d, pipeline.TakeContext(long.chunks, excess, long=long)


def text_of(script):
    return " ".join(w for w, _, _ in script)


def test_short_take_is_transcribed_whole_as_before(monkeypatch):
    d = make_dictation(monkeypatch)
    model = d.models.model = ScriptModel(speech(10))
    d.toggle()
    assert d.long is not None
    d.chunks.clear()  # just the ramp below, not the fake mic's first second
    for i in range(0, 10 * RATE, RATE):
        d._capture(ramp(10)[i:i + RATE, None])
        d.long.tick()
    d.toggle()
    assert [(c.start, c.end, c.words) for c in model.calls] == [(0, 10 * RATE, False)]
    assert d.pasted == [text_of(speech(10))]


def test_long_take_passes_while_recording_and_finishes_the_rest_at_the_stop(monkeypatch):
    script = speech(50, pause_every=8)
    d, take = long_take(monkeypatch, ScriptModel(script), 50)
    long = take.long
    assert long.passes > 5 and long.words and not long.failed
    passes = len(d.models.model.calls)
    assert all(c.words and not c.fallback for c in d.models.model.calls)  # no slow retries mid-take
    assert d.models.model.calls[0].start == 0 and d.models.model.calls[0].end >= 20 * RATE  # not before 20 s
    d.pipeline.run(take)
    last = d.models.model.calls[passes:]
    assert [(c.start, c.end, c.fallback) for c in last] == [(long.pass_start(), 50 * RATE, True)]  # only the rest
    assert d.pasted == [text_of(script)]


def test_agreement_is_the_common_prefix_ignoring_case_and_punctuation():
    a = [("Hello,", 0, 1), ("world", 1, 2), ("again", 2, 3)]
    b = [(" hello", 0, 1), ("World.", 1, 2), ("gain", 2, 3), ("again", 3, 4)]
    assert streaming.agreement(a, b) == 2
    assert streaming.agreement([], b) == 0


def test_only_agreed_words_are_kept_and_the_cut_is_at_a_word_end(monkeypatch):
    d = make_dictation(monkeypatch)
    long = LongTake(d.models, [], store.settings())
    long.hypothesis = [(" a", 0, 100), (" b", 110, 200), (" c", 210, 300)]
    long.commit([(" A.", 0, 100), (" b", 110, 205), (" x", 215, 300), (" d", 310, 400)],
                streaming.agreement(long.hypothesis, [(" A.", 0, 0), (" b", 0, 0), (" x", 0, 0)]))
    assert [w for w, _, _ in long.words] == [" A.", " b"]  # the later pass's spelling
    assert long.committed == 205
    assert [w for w, _, _ in long.hypothesis] == [" x", " d"]


def test_cut_prefers_the_pause_after_an_agreed_word(monkeypatch):
    d = make_dictation(monkeypatch)
    long = LongTake(d.models, [], store.settings())
    gap = int(streaming.PAUSE * RATE)
    words = [(" a", 0, 100), (" b", 100 + gap, 200 + gap), (" c", 210 + gap, 300 + gap)]
    long.commit(words, 3)
    assert [w for w, _, _ in long.words] == [" a"] and long.committed == (100 + 100 + gap) // 2
    assert long.hypothesis == words[1:]  # agreed again on the next pass


def test_a_word_corrected_by_a_later_pass_is_pasted_corrected(monkeypatch):
    script = speech(40)
    misheard = script[45][0]  # heard wrong while near the end of what's recorded
    model = ScriptModel(script, early={misheard: "WRONG"})
    d, take = long_take(monkeypatch, model, 40)
    d.pipeline.run(take)
    assert d.pasted == [text_of(script)] and "WRONG" not in d.pasted[0]


def test_misheard_last_words_are_redone_by_the_last_pass(monkeypatch):
    script = speech(30)
    model = ScriptModel(script, early={w: "WRONG" for w, _, _ in script[-3:]})
    d, take = long_take(monkeypatch, model, 30.6)
    assert take.long.committed < int(script[-3][1] * RATE)  # still provisional at the stop
    d.pipeline.run(take)
    assert d.pasted == [text_of(script)]


class LateModel(ScriptModel):
    """Gives every word's timestamps 0.2 s late, as Whisper can: a cut at a word's reported
    end falls inside the next word."""
    def transcribe(self, audio, **kw):
        segments, info = super().transcribe(audio, **kw)
        for w in segments[0].words:
            w.start, w.end = w.start + 0.2, w.end + 0.2
        return segments, info


def test_word_after_a_cut_with_late_timestamps_is_not_clipped(monkeypatch):
    script = speech(40)  # no pauses: every cut is at a word's reported end
    d, take = long_take(monkeypatch, LateModel(script), 40)
    assert take.long.passes and take.long.words
    d.pipeline.run(take)
    assert d.pasted == [text_of(script)]  # nothing lost at the cuts, and nothing heard twice


class FickleModel(ScriptModel):
    """Never hears the same words twice: no pass agrees with the one before."""
    def transcribe(self, audio, **kw):
        segments, info = super().transcribe(audio, **kw)
        for w in segments[0].words:
            w.word += str(len(self.calls))
        return segments, info


def test_window_stays_within_whisper_s_30_seconds_without_agreement(monkeypatch):
    model = FickleModel(speech(70))
    _, take = long_take(monkeypatch, model, 70)
    assert not take.long.failed and take.long.words
    assert max(c.end - c.start for c in model.calls) <= streaming.MAX_WINDOW * RATE


def test_long_silence_gives_up_and_transcribes_the_take_whole(monkeypatch):
    model = ScriptModel([])
    d, take = long_take(monkeypatch, model, 40)
    assert take.long.failed
    d.pipeline.run(take)
    assert (model.calls[-1].start, model.calls[-1].end, model.calls[-1].words) == (0, 40 * RATE, False)


@pytest.mark.parametrize("fail_at", [1, 3])
def test_failed_pass_falls_back_to_the_whole_take(monkeypatch, fail_at):
    script = speech(40)
    model = ScriptModel(script, fail_at=fail_at)
    d, take = long_take(monkeypatch, model, 40)
    assert take.long.failed and len(model.calls) == fail_at  # no passes after a failure
    d.pipeline.run(take)
    assert (model.calls[-1].start, model.calls[-1].end, model.calls[-1].words) == (0, 40 * RATE, False)
    assert d.pasted == [text_of(script)]


def test_failed_last_pass_falls_back_to_the_whole_take(monkeypatch):
    script = speech(40)
    model = ScriptModel(script)
    d, take = long_take(monkeypatch, model, 40)
    model.fail_at = len(model.calls) + 1
    d.pipeline.run(take)
    assert (model.calls[-1].start, model.calls[-1].end, model.calls[-1].words) == (0, 40 * RATE, False)
    assert d.pasted == [text_of(script)]


def test_no_model_loaded_falls_back_to_the_whole_take(monkeypatch):
    model = ScriptModel(speech(30))
    model.is_loaded = False
    _, take = long_take(monkeypatch, model, 30)
    assert take.long.failed and model.calls == []


def test_language_of_the_first_pass_is_kept(monkeypatch):
    model = ScriptModel(speech(40), lang="fr")
    d, take = long_take(monkeypatch, model, 40)
    d.pipeline.run(take)
    assert model.calls[0].language is None  # detected on the first pass
    assert {c.language for c in model.calls[1:]} == {"fr"}  # every later pass and the last one
    assert d.models.last_lang == "fr"


def test_set_language_is_used_from_the_first_pass(monkeypatch):
    store.save_settings({"language": "de"})
    model = ScriptModel(speech(30), lang="fr")
    long_take(monkeypatch, model, 30)
    assert {c.language for c in model.calls} == {"de"}


def test_later_passes_are_prompted_with_the_end_of_the_text_so_far(monkeypatch):
    model = ScriptModel([(f"word{i}", s, e) for i, (_, s, e) in enumerate(speech(70))])
    d, take = long_take(monkeypatch, model, 70)
    d.pipeline.run(take)
    assert model.calls[0].prompt == engine.PROMPT  # the first, as for a whole take
    later = [c.prompt for c in model.calls[1:]]
    assert any(len(p) > 150 for p in later)
    assert all(len(p) <= streaming.CONTEXT_CHARS and p.startswith("word") for p in later if p != engine.PROMPT)
    assert later[-1].endswith(streaming.joined(take.long.words)[-20:])


def test_excess_after_the_release_is_dropped_from_the_last_pass(monkeypatch):
    script = speech(40)
    model = ScriptModel(script)
    d, take = long_take(monkeypatch, model, 40.5, excess=RATE // 2)
    d.pipeline.run(take)
    assert model.calls[-1].end == 40 * RATE
    assert d.pasted == [text_of(script)]


def test_off_switch_transcribes_long_takes_whole(monkeypatch):
    store.save_settings({"stream_long_takes": False})
    d = make_dictation(monkeypatch)
    d.models.model = ScriptModel(speech(30))
    d.toggle()
    assert d.long is None
    d.toggle()


def test_engine_without_word_timestamps_does_not_stream(monkeypatch):
    d = make_dictation(monkeypatch)  # FakeModel, like whisper.cpp, has no word timestamps
    d.toggle()
    assert d.long is None
    d.toggle()
    assert d.pasted == ["take1"]


def test_stream_setting_follows_the_environment_and_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("ASR_STREAM", "0")
    assert store.settings().stream_long_takes is False
    monkeypatch.delenv("ASR_STREAM")
    with open(store.CONFIG, "w") as f:
        f.write('stream_long_takes = "maybe"\n')
    assert store.settings().stream_long_takes is True


def test_take_stopped_mid_recording_stops_its_passes(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = ScriptModel(speech(30))
    monkeypatch.setattr(streaming, "TICK", 0.01)
    d.toggle()
    long = d.long
    d.toggle()
    long.thread.join(2)
    assert not long.thread.is_alive() and d.long is None


def test_timing_line_counts_the_passes(monkeypatch, caplog):
    d, take = long_take(monkeypatch, ScriptModel(speech(40)), 40)
    with caplog.at_level("INFO", logger="tuxwhisper"):
        d.pipeline.run(take)
    assert any("passes while speaking" in r.message and "s tail" in r.message for r in caplog.records)
