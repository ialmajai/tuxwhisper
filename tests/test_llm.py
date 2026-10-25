"""Tests for the LLM cleanup."""

import pytest
from conftest import notified

from tuxwhisper import llm, store


@pytest.mark.parametrize("minutes, keep_alive", [(30, "30m"), (0, "0m"), (-1, -1)])
def test_llm_keep_alive(minutes, keep_alive):
    with open(store.CONFIG, "w") as f:
        f.write(f"llm_unload_minutes = {minutes}\n")
    assert llm.llm_keep_alive() == keep_alive


@pytest.mark.parametrize("minutes", [0, 30])
def test_loading_keeps_the_model_for_the_cleanup_even_when_unloading_right_after_use(monkeypatch, minutes):
    with open(store.CONFIG, "w") as f:
        f.write(f"llm_unload_minutes = {minutes}\n")
    loaded = []
    def ollama(path, payload=None, timeout=30):  # like Ollama: keep_alive "0m" unloads at once
        if path == "/api/ps":
            return {"models": [{"name": "llama3.2:latest", "size": 2, "size_vram": 2}] if loaded else []}
        if payload.get("keep_alive") not in (0, "0m"):
            loaded.append(payload["keep_alive"])
        return {}
    monkeypatch.setattr(llm, "ollama", ollama)
    assert llm.rewrite_model_on_gpu()  # loaded and on the GPU, not "not enough GPU memory"
    assert loaded == ["5m" if minutes == 0 else f"{minutes}m"]


@pytest.mark.parametrize("text, cued", [
    ("let's meet on tuesday no wait wednesday", True),
    ("send it to john sorry i mean mike", True),
    ("at four actually make that five", True),
    ("call you tomorrow or rather friday", True),
    ("book six scratch that eight", True),
    ("report dot txt no no report dot md", True),
    ("what i mean is that the tests are flaky", False),
    ("is it tuesday or wednesday", False),
    ("there are no errors in the log", False),
    ("i actually liked the first version", False),
    ("send it to John, sorry, Jim.", True),
    ("Meet at four, make that five.", True),
    ("Tuesday, I meant Wednesday.", True),
    ("Can you make that call tomorrow?", False),
    ("I meant to tell you about the release.", False),
    ("Tell her I'm sorry about the delay.", False),
    ("I mean it, the tests are flaky.", False),
])
def test_self_correction_rule_only_for_correction_cues(text, cued):
    assert (llm.SELF_CORRECTION in llm.rewrite_task(text)) is cued
    assert llm.rewrite_task(text).startswith(llm.modes()[llm.current_mode()])


def test_context_size_grows_with_the_transcript():
    assert llm.context_size("x" * 1500, "word " * 50) == 4096
    assert llm.context_size("x" * 1500, "word " * 700) == 4096
    assert llm.context_size("x" * 1500, "word " * 1100) == 8192
    assert llm.context_size("x" * 1500, "word " * 3000) is None


@pytest.mark.parametrize("reply", [
    {"done_reason": "length", "prompt_eval_count": 10, "eval_count": 10},
    {"done_reason": "stop", "prompt_eval_count": 4070, "eval_count": 30},  # input was trimmed
])
def test_cleanup_that_ran_out_of_room_pastes_the_plain_transcript(monkeypatch, reply):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda path, payload, timeout=30: {
        **reply, "message": {"content": '{"cleaned": "half of it"}'}})
    assert llm.llm_cleanup("um the whole take", "en") == "um the whole take"
    assert notified == ["Too long for LLM cleanup; pasted the plain transcript"]


def test_cleanup_without_token_counts_is_kept(monkeypatch):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda path, payload, timeout=30: {
        "done_reason": "stop", "prompt_eval_count": 0, "message": {"content": '{"cleaned": "The take."}'}})
    assert llm.llm_cleanup("um the take", "en") == "The take."


def test_far_too_long_take_skips_cleanup(monkeypatch):
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", lambda: True)
    monkeypatch.setattr(llm, "ollama", lambda *a, **k: pytest.fail("should not be called"))
    text = "word " * 5000
    assert llm.llm_cleanup(text, "en") == text


def test_warm_up_and_cleanup_use_the_same_context():
    assert llm.OLLAMA_OPTIONS["num_ctx"] == llm.context_size("x" * 1500, "word " * 300)


def test_cleanup_trusts_the_warm_up(monkeypatch):
    def checked():
        raise AssertionError("checked the GPU again")
    monkeypatch.setattr(llm, "rewrite_model_on_gpu", checked)
    monkeypatch.setattr(llm, "ollama", lambda path, payload, timeout=30: {
        "done_reason": "stop", "message": {"content": '{"cleaned": "The take."}'}})
    assert llm.llm_cleanup("um the take", "en", on_gpu=True) == "The take."


def test_modes_file_skips_comments_and_half_written_lines():
    with open(store.MODES_FILE, "w") as f:
        f.write("  # Indented => comment\n=> no name\nNo task =>\nTweet => Rewrite as a tweet.\n")
    found = llm.modes()
    assert found["Tweet"] == "Rewrite as a tweet."
    assert not {"# Indented", "", "No task"} & found.keys()


@pytest.mark.parametrize("model, listed", [
    ("llama3.2", "llama3.2:latest"),
    ("qwen3:8b", "qwen3:8b"),
    ("localhost:5000/team/m", "localhost:5000/team/m:latest"),  # a registry's port isn't a tag
])
def test_gpu_check_finds_the_model_ollama_lists(monkeypatch, model, listed):
    with open(store.CONFIG, "w") as f:
        f.write(f'rewrite_model = "{model}"\n')
    def ollama(path, payload=None, timeout=30):
        assert path == "/api/ps", "already loaded: nothing to load or unload"
        return {"models": [{"name": listed, "size": 2, "size_vram": 2}]}
    monkeypatch.setattr(llm, "ollama", ollama)
    assert llm.rewrite_model_on_gpu()


def test_context_size_allows_for_scripts_that_take_more_tokens_a_character():
    system = "x" * 1500
    assert llm.context_size(system, "word " * 600) == 4096   # 3000 letters of English
    assert llm.context_size(system, "字" * 3000) == 8192     # about a token a character
