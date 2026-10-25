"""Tests for the dictation core: takes, pasting, saving and the socket server."""
import os
import stat
import threading
import time
import types

import numpy as np
import pytest
from conftest import (
    FakeKeyboard,
    FakeStream,
    InlineThread,
    OOMModel,
    RunOOMModel,
    WhisperedModel,
    burst,
    fake_gpu,
    make_dictation,
    notified,
    real_add_history,
)

from tuxwhisper import daemon, desktop, engine, llm, pipeline, recordings, server, store


def test_overlapping_takes_keep_their_own_mode(monkeypatch):
    """A cleanup take still transcribing must not switch to raw when a plain take starts."""
    d = make_dictation(monkeypatch)
    d.models.model.release_first.clear()
    d.toggle(rewrite=True)                     # take 1: cleanup
    first = threading.Thread(target=d.toggle)  # stop take 1; it transcribes slowly
    first.start()
    assert d.models.model.first_started.wait(5)
    d.toggle(rewrite=False)                    # take 2: plain, starts while take 1 transcribes
    second = threading.Thread(target=d.toggle)
    second.start()
    time.sleep(0.2)
    d.models.model.release_first.set()
    first.join(5)
    second.join(5)
    assert d.pasted == ["CLEANED:take1", "take2"]  # right mode for each, in order


def test_microphone_failure_does_not_wedge(monkeypatch):
    d = make_dictation(monkeypatch)
    FakeStream.fail_start = True
    d.toggle()
    assert d.stream is None
    assert any("microphone" in m.lower() for m in notified)
    FakeStream.fail_start = False
    d.toggle()                                 # the next press records normally
    d.toggle()
    assert d.pasted == ["take1"]


def save_with_answer(monkeypatch, d, answer):
    """Press the save key; the confirmation dialog returns answer (None = Cancel)."""
    shown = []
    monkeypatch.setattr(recordings, "threading", types.SimpleNamespace(Thread=InlineThread))
    monkeypatch.setattr(desktop, "edit_text", lambda text, title, ok, explain=None: shown.append(text) or answer)
    d.save()
    return shown


def saved_rows():
    with open(os.path.join(store.RECORDINGS, "metadata.csv")) as f:
        return f.read().splitlines()[1:]


def test_saved_recordings_are_private(monkeypatch):
    d = make_dictation(monkeypatch)
    d.recordings.last_take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "hello")
    save_with_answer(monkeypatch, d, "hello")
    mode = stat.S_IMODE(os.stat(store.RECORDINGS).st_mode)
    assert mode == 0o700


def test_save_asks_first_and_keeps_the_fixed_transcript(monkeypatch):
    d = make_dictation(monkeypatch)
    d.recordings.last_take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "send it to jon")
    shown = save_with_answer(monkeypatch, d, "send it to John")
    assert shown == ["send it to jon"]
    assert [r.split(",", 1)[1] for r in saved_rows()] == ["send it to John"]


def test_save_dialog_that_fails_is_reported_and_the_take_kept(monkeypatch):
    d = make_dictation(monkeypatch)
    take = d.recordings.last_take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "hello")
    monkeypatch.setattr(recordings, "threading", types.SimpleNamespace(Thread=InlineThread))
    def missing(*a):
        raise RuntimeError("zenity isn't installed")
    monkeypatch.setattr(desktop, "edit_text", missing)
    d.save()
    assert notified == ["Couldn't open the save dialog"] and d.recordings.last_take is take


def test_save_that_fails_to_write_is_reported_and_the_take_kept(monkeypatch):
    d = make_dictation(monkeypatch)
    take = d.recordings.last_take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "hello")
    def full(*a):
        raise OSError("No space left on device")
    d.recordings.write_recording = full
    save_with_answer(monkeypatch, d, "hello")
    assert notified == ["Couldn't save the recording"] and d.recordings.last_take is take


@pytest.mark.parametrize("err, body", [
    (OSError(28, "No space left on device"), "The disk is full."),
    (PermissionError(13, "Permission denied"), "No permission to write to {}."),
    (ValueError("odd"), "odd")])
def test_save_errors_are_said_plainly(err, body):
    assert recordings.write_error(err) == body.format(store.RECORDINGS)


def test_cancelled_save_does_not_bring_back_an_older_take(monkeypatch):
    d = make_dictation(monkeypatch)
    d.recordings.last_take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "old")
    monkeypatch.setattr(recordings, "threading", types.SimpleNamespace(Thread=InlineThread))
    def dialog(text, title, ok, explain=None):
        # While it's open, a newer take is dictated and the save key takes it for its own dialog.
        d.recordings.takes += 1
        d.recordings.last_take = None
        # ... and then this one is cancelled (returns None)
    monkeypatch.setattr(desktop, "edit_text", dialog)
    d.save()
    assert d.recordings.last_take is None


def test_cancelled_save_writes_nothing_and_can_be_asked_again(monkeypatch):
    d = make_dictation(monkeypatch)
    take = recordings.Take(np.zeros(store.RATE, dtype=np.float32), "hello")
    d.recordings.last_take = take
    save_with_answer(monkeypatch, d, None)
    assert not os.path.exists(store.RECORDINGS)
    assert d.recordings.last_take is take  # the save key offers it again


def test_speech_model_unloads_when_idle_and_reloads_on_use(monkeypatch):
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")
    d = make_dictation(monkeypatch)
    d.unload_if_idle()
    assert d.models.model.loaded()                   # used just now: stays loaded
    d.models.last_used -= 6 * 60
    d.unload_if_idle()
    assert not d.models.model.loaded()               # 6 min idle: freed
    d.toggle()                                # the next take reloads it before transcribing
    d.toggle()
    assert d.models.model.loaded() and d.pasted == ["take1"]


def test_speech_model_kept_when_never_or_cpu(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.last_used -= 120 * 60
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 0\n")    # never
    d.unload_if_idle()
    assert d.models.model.loaded()
    monkeypatch.setattr(engine, "DEVICE", "cpu")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")    # no GPU memory to free on the CPU
    d.unload_if_idle()
    assert d.models.model.loaded()


def test_no_unload_while_transcribing(monkeypatch):
    monkeypatch.setattr(engine, "DEVICE", "cuda")
    with open(store.CONFIG, "w") as f:
        f.write("asr_unload_minutes = 5\n")
    d = make_dictation(monkeypatch)
    d.models.last_used -= 6 * 60
    with d.models.lock:                        # a take is being transcribed
        d.unload_if_idle()
    assert d.models.model.loaded()


def test_reload_out_of_memory_falls_back_and_still_pastes(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = OOMModel()
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    d.toggle()
    d.toggle()
    assert loads == ["small"]
    assert d.pasted == ["take1"]
    assert d.models.fallback_from == "large-v3-turbo"
    assert any("using small" in n for n in notified)


def test_no_model_fits_notifies_once_and_skips_the_take(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = OOMModel()
    fake_gpu(monkeypatch, [0.05], [])
    d.toggle()
    d.toggle()
    assert d.pasted == []
    assert sum("Not enough GPU memory" in n for n in notified) == 1


def test_switches_back_when_memory_frees(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = OOMModel()
    free, loads = [1.2], []
    fake_gpu(monkeypatch, free, loads)
    d.toggle()
    d.toggle()
    free[0] = 4.0
    d.restore_preferred_model()  # the idle thread does this between takes
    assert loads == ["small", engine.asr_model()]
    assert d.models.fallback_from is None
    assert any("Back to large-v3-turbo" in n for n in notified)


def test_take_loads_a_model_once_there_is_room(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model, d.models.fallback_from = None, "large-v3-turbo"
    free, loads = [0.05], []
    fake_gpu(monkeypatch, free, loads)
    d.toggle()
    d.toggle()
    assert d.pasted == [] and loads == []
    free[0] = 1.2
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.pasted == ["take1"]


def test_retype_erases_the_dictation_and_pastes_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.paster.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.paster.kbd = FakeKeyboard()
    with open(store.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert d.retype()
    assert d.paster.kbd.keys == [14] * 6 and d.pasted == ["Hello"]
    assert not os.path.exists(store.retype_path())


def test_typed_length_counts_characters_not_accents():
    assert daemon.typed_length("café") == 4
    assert daemon.typed_length("café") == 4


def test_failed_take_is_reported_and_the_next_one_works(monkeypatch):
    d = make_dictation(monkeypatch)
    calls = []
    def transcribe(audio, **_):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return [types.SimpleNamespace(text="fine")], None
    d.models.model.transcribe = transcribe
    d.toggle()
    d.toggle()
    assert "Dictation failed" in notified and d.pasted == []
    d.toggle()
    d.toggle()
    assert d.pasted == ["fine"]


def test_mic_error_on_stop_does_not_wedge(monkeypatch):
    d = make_dictation(monkeypatch)
    d.toggle()
    def broken():
        raise OSError("device unplugged")
    d.stream.stop = broken
    d.toggle()
    assert d.stream is None


def test_retype_refuses_while_a_take_is_in_flight(monkeypatch):
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    d.paster.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.paster.kbd = FakeKeyboard()
    d.next_ticket = 1  # a take stopped but isn't pasted yet
    with open(store.retype_path(), "w") as f:
        f.write('{"delete": 6, "text": "Hello"}')
    assert not d.retype() and d.paster.kbd.keys == []


def test_history_records_whether_the_take_was_typed(monkeypatch):
    d = make_dictation(monkeypatch)
    monkeypatch.setattr(store, "add_history", real_add_history)
    d.paster.paste = lambda text, snapshot=None: False  # e.g. the clipboard couldn't be set
    d.toggle()
    d.toggle()
    assert store.load_history()[-1]["typed"] is False


def test_press_right_after_a_hold_ends_starts_a_new_take(monkeypatch):
    d = make_dictation(monkeypatch)
    monkeypatch.setattr(d.models, "can_stream", lambda: False)
    clock = [100.0]
    monkeypatch.setattr(daemon.time, "monotonic", lambda: clock[0])

    def until(cond):
        deadline = time.time() + 5
        while not cond() and time.time() < deadline:
            time.sleep(0.01)
        assert cond()

    d.press()  # the key goes down: a take starts
    until(lambda: d.stream is not None)
    for t in (100.5, 100.6, 100.7):  # GNOME's auto-repeat while the key is held
        clock[0] = t
        d.press()
    assert d.held
    clock[0] = 101.1  # no repeat for RELEASE_GAP: the key was let go
    d.check_release()
    until(lambda: d.stream is None)
    clock[0] = 101.3  # pressed again within REPEAT_WINDOW of the last repeat
    d.press()
    until(lambda: d.stream is not None)  # a new take, not a repeat of the old one
    d.toggle()


def test_start_beep_waits_for_the_mic_s_first_audio(monkeypatch):
    d = make_dictation(monkeypatch)
    beeps = []
    monkeypatch.setattr(desktop, "sound", lambda name: beeps.append((name, len(d.chunks))))
    d.toggle()
    assert beeps == [("device-added", 1)]  # not before audio arrives: words after it are kept


def test_stop_before_recording_notice_closes_it(monkeypatch):
    d = make_dictation(monkeypatch)
    closed = []
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    def slow_notify(msg, **kw):
        if msg == "Recording…":
            d.toggle()  # the take stops while the notice is still being shown
            return "42"
    monkeypatch.setattr(desktop, "notify", slow_notify)
    d.toggle()
    assert "42" in closed and d.indicator is None


def test_recording_notice_becomes_transcribing_and_is_closed(monkeypatch):
    d = make_dictation(monkeypatch)
    shown, closed = [], []
    def fake_notify(msg, **kw):
        shown.append((msg, kw.get("replace")))
        return "7" if kw.get("persistent") else None
    monkeypatch.setattr(desktop, "notify", fake_notify)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    d.toggle()
    d.toggle()
    assert shown[:2] == [("Recording…", None), ("Transcribing…", "7")]  # replaced in place
    assert closed == ["7"] and d.pasted


def test_transcribing_notice_is_closed_when_a_take_fails(monkeypatch):
    d = make_dictation(monkeypatch)
    closed = []
    monkeypatch.setattr(desktop, "notify", lambda msg, **kw: "7" if kw.get("persistent") else None)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    def boom(*a, **k):
        raise RuntimeError("boom")
    d.models.model.transcribe = boom
    d.toggle()
    d.toggle()
    assert closed == ["7"]


def test_slow_cleanup_replaces_transcribing_in_place(monkeypatch):
    d = make_dictation(monkeypatch)
    shown, closed = [], []
    def fake_notify(msg, **kw):
        shown.append((msg, kw.get("replace")))
        return "7" if kw.get("persistent") else None
    monkeypatch.setattr(desktop, "notify", fake_notify)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    monkeypatch.setattr(daemon.threading, "Timer", lambda delay, fn: types.SimpleNamespace(
        start=fn, cancel=lambda: None, join=lambda: None))  # the cleanup is slow: fires at once
    d.toggle(rewrite=True)
    d.toggle()
    assert [m for m in shown if m[0].startswith("Cleaning up")] == [(f"Cleaning up ({llm.current_mode()})…", "7")]
    assert closed == ["7"]


def test_no_model_switch_while_recording(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.fallback_from = "large-v3-turbo"
    loads = []
    fake_gpu(monkeypatch, [4.0], loads)
    d.stream = object()
    d.restore_preferred_model()
    assert loads == []


@pytest.mark.parametrize("seconds, text, title", [
    (0.5, "", None),                                     # a stray key press
    (2.7, "", "Didn't catch that"),
    (20.8, "How would it bother how would?", "Only caught a few words"),
    (20.0, " ".join(["word"] * 40), None),               # normal speech with pauses
    (6.0, "Yes.", None),                                 # short takes can be short
])
def test_missed_speech(seconds, text, title):
    missed = pipeline.missed_speech(seconds, text)
    assert (missed[0] if missed else None) == title


def test_empty_take_is_reported(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model.transcribe = lambda audio, **_: ([], None)
    d.toggle()
    d.toggle()
    assert "Didn't catch that" in notified and d.pasted == []


@pytest.mark.parametrize("heard, sound, typed", [
    ("Yes.", 0.4, ["Yes."]),
    ("God bless you.", 0.3, []),
])
def test_whispered_take_is_retried_without_the_voice_filter(monkeypatch, heard, sound, typed):
    d = make_dictation(monkeypatch)
    d.models.model = WhisperedModel(heard)
    d.pipeline.run(pipeline.TakeContext([burst(sound)[:, None]]))
    assert d.pasted == typed
    assert ("Didn't catch that" in notified) is (not typed)


@pytest.mark.parametrize("without_prompt, typed", [
    ("Option one.", ["Option one."]),
    ("", []),
])
def test_made_up_sentence_is_retried_without_the_prompt(monkeypatch, without_prompt, typed):
    d = make_dictation(monkeypatch)
    made_up = "I'm going to start with a sentence, with punctuation and capitals."
    d.models.model.transcribe = lambda audio, prompt=None, **_: (
        [types.SimpleNamespace(text=made_up if prompt else without_prompt)], None)
    d.pipeline.run(pipeline.TakeContext([burst(0.8)[:, None]]))
    assert d.pasted == typed
    assert ("Didn't catch that" in notified) is (not typed)


def test_fast_speech_is_not_retried_without_the_prompt(monkeypatch):
    d = make_dictation(monkeypatch)
    said = "I'll repeat the repository till the very end."  # 15 counted syllables in 1.7 s
    d.models.model.transcribe = lambda audio, prompt=None, **_: (
        [types.SimpleNamespace(text=said if prompt else "repository till the very end.")], None)
    d.pipeline.run(pipeline.TakeContext([burst(1.7)[:, None]]))
    assert d.pasted == [said]


def test_silence_is_not_retried(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = WhisperedModel("You")
    d.pipeline.run(pipeline.TakeContext([np.zeros((4 * 16000, 1), dtype=np.float32)]))
    assert d.pasted == [] and "Didn't catch that" in notified


def test_take_that_runs_out_of_gpu_memory_switches_down_and_still_pastes(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = RunOOMModel()
    loads = []
    fake_gpu(monkeypatch, [1.2], loads)
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.pasted == ["take1"]
    assert any("using small" in n for n in notified)


def test_take_that_runs_out_of_memory_again_after_switching_down_is_reported(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = RunOOMModel()
    fake_gpu(monkeypatch, [1.2], [])
    monkeypatch.setattr(engine, "load_whisper", lambda name: RunOOMModel())  # loads, then can't run
    d.toggle()
    d.toggle()
    assert d.pasted == [] and notified[-1] == "Not enough GPU memory to dictate"


def test_take_with_no_model_that_fits_shows_only_the_memory_notice(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = RunOOMModel()
    fake_gpu(monkeypatch, [0.05], [])
    d.toggle()
    d.toggle()
    assert "Not enough GPU memory to dictate" in notified and "Didn't catch that" not in notified


def test_out_of_memory_steps_down_from_the_model_in_use(monkeypatch):
    d = make_dictation(monkeypatch)
    loads = []
    fake_gpu(monkeypatch, [1.6], loads)
    d.models.model, d.models.name, d.models.fallback_from = RunOOMModel(), "medium", "large-v3-turbo"
    d.toggle()
    d.toggle()
    assert loads == ["small"] and d.models.name == "small"  # medium would fit, but just failed
    assert d.models.fallback_from == "large-v3-turbo"  # still back to the chosen model once there's room


def test_long_take_is_not_retried_without_the_voice_filter(monkeypatch):
    d = make_dictation(monkeypatch)
    d.models.model = WhisperedModel("Yes.")
    audio = np.concatenate([burst(0.4), np.zeros(28 * store.RATE, dtype=np.float32)])
    d.pipeline.run(pipeline.TakeContext([audio[:, None]]))
    assert d.pasted == []


def test_transcribing_notice_does_not_hold_up_the_take(monkeypatch):
    d = make_dictation(monkeypatch)
    shown, closed = threading.Event(), []
    def slow_notify(msg, **kw):
        if msg == "Transcribing…":
            assert d.models.model.first_started.wait(5)  # the take transcribes while this is shown
            shown.set()
            return "7"
    monkeypatch.setattr(desktop, "notify", slow_notify)
    monkeypatch.setattr(desktop, "close_notification", lambda nid: closed.append(nid))
    d.toggle()
    d.toggle()
    assert shown.is_set() and closed == ["7"] and d.pasted == ["take1"]


def test_paste_restores_the_clipboard_read_during_transcription(monkeypatch):
    monkeypatch.setattr(daemon.time, "sleep", lambda s: None)
    d = make_dictation(monkeypatch)
    del d.paster.paste  # the real one
    d.paster.ec = types.SimpleNamespace(EV_KEY=1, KEY_LEFTCTRL=29, KEY_V=47, KEY_LEFTSHIFT=42, KEY_INSERT=110)
    d.paster.kbd = FakeKeyboard()
    board = {"clipboard": "old", "primary": "old primary"}
    monkeypatch.setattr(desktop, "clipboard_get", lambda sel="clipboard": board[sel])
    sets = []
    monkeypatch.setattr(desktop, "clipboard_set", lambda text, sel="clipboard": sets.append((sel, text)) or True)
    store.save_settings({"paste_key": "ctrl+v"})
    snapshot = d.paster.snapshot_clipboard()
    store.save_settings({"paste_key": "shift+insert"})  # changed mid-take: primary is read at paste
    assert d.paster.paste("hi ", snapshot)
    assert sorted(sets) == sorted([("clipboard", "hi "), ("primary", "hi "),
                                   ("clipboard", "old"), ("primary", "old primary")])
    assert sets.index(("clipboard", "old")) > sets.index(("clipboard", "hi "))


def test_take_passes_the_warm_up_result_to_the_cleanup(monkeypatch):
    d = make_dictation(monkeypatch)
    monkeypatch.setattr(llm, "warm_rewrite_model", lambda: True)
    got = []
    monkeypatch.setattr(llm, "llm_cleanup", lambda text, lang, on_gpu=None: got.append(on_gpu) or text)
    d.toggle(rewrite=True)
    d.toggle()
    assert got == [True]


def test_connection_command_runs(monkeypatch):
    import socket
    d = types.SimpleNamespace(presses=[])
    d.press = lambda rewrite=False: d.presses.append(rewrite)
    a, b = socket.socketpair()
    b.sendall(b"rewrite")
    b.shutdown(socket.SHUT_WR)
    server.handle(d, a)
    assert d.presses == [True]


def test_take_stopped_during_a_retype_pastes_after_it(monkeypatch):
    d = make_dictation(monkeypatch)
    d.paster.ec = types.SimpleNamespace(EV_KEY=1, KEY_BACKSPACE=14)
    d.paster.kbd = FakeKeyboard()
    order = []
    typing = threading.Event()
    def paste(text, snapshot=None):
        if text == "Hello ":
            typing.set()
            time.sleep(0.2)
        order.append(text.strip())
        return True
    d.paster.paste = paste
    with open(store.retype_path(), "w") as f:
        f.write('{"delete": 1, "text": "Hello"}')
    retyping = threading.Thread(target=d.retype)
    retyping.start()
    assert typing.wait(5)
    d.toggle()
    d.toggle()
    retyping.join()
    assert order == ["Hello", "take1"]


def test_settings_key_is_looked_up_at_most_every_few_minutes(monkeypatch):
    d = make_dictation(monkeypatch)
    looked = []
    d.refresh_hint = lambda: looked.append(1)
    monkeypatch.setattr(daemon.threading, "Thread", InlineThread)
    for _ in range(3):
        d.pipeline.run(pipeline.TakeContext([burst(1)[:, None]]))
    assert looked == [1]


# --- a stuck cleanup, lost audio, and the take's timing ---------------------------------------

def test_stuck_cleanup_pastes_the_plain_transcript_after_the_deadline(monkeypatch):
    d = make_dictation(monkeypatch)
    release = threading.Event()
    monkeypatch.setattr(llm, "llm_cleanup", lambda text, lang, on_gpu=None: release.wait(5) and "late")
    monkeypatch.setattr(pipeline, "cleanup_deadline", lambda text: 0.2)
    d.toggle(rewrite=True)
    d.toggle()
    release.set()
    assert d.pasted == ["take1"]  # not held up, not cleaned
    assert "LLM cleanup took too long; pasted the plain transcript" in notified


def test_cleanup_deadline_grows_with_the_text():
    assert pipeline.cleanup_deadline("short") == 15
    assert pipeline.cleanup_deadline("x" * 2000) == 25


def test_lost_audio_is_counted_and_reported(monkeypatch):
    d = make_dictation(monkeypatch)
    d.toggle()
    d._capture(np.zeros((160, 1), dtype=np.float32), 160, None, types.SimpleNamespace(input_overflow=True))
    d.toggle()
    assert "Some audio was lost while recording" in notified
    assert d.pasted == ["take1"]  # still transcribed and pasted


def test_take_logs_its_stage_timings(monkeypatch, caplog):
    d = make_dictation(monkeypatch)
    with caplog.at_level("INFO", logger="tuxwhisper"):
        d.toggle()
        d.toggle()
    assert any("timing: transcribe" in r.message and "from stop to pasted" in r.message for r in caplog.records)
