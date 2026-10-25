"""The settings in use, as one typed value: kept out of store, which every key press imports."""
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from tuxwhisper.store import LANGUAGES, PASTE_KEYS


@dataclass(frozen=True)
class Settings:
    """The settings in use: environment defaults (ASR_*) overridden by the config file."""
    sounds: bool = True  # beep on start and stop
    # Percent of the system volume, so the beeps never get louder than your other sounds.
    sound_volume: int = 40
    language: str = ""  # e.g. "en"; "" = auto-detect
    # Shift+Insert pastes in browsers, editors and terminals alike, and unlike Ctrl+V it doesn't
    # depend on the keyboard layout (keys are sent by position, so Ctrl+V is Ctrl+K on Dvorak).
    paste_key: str = "shift+insert"
    rewrite_model: str = "llama3.2"
    mode: str = "Clean up"
    asr_model: str = ""  # "" = the engine's default
    # Minutes idle before freeing GPU memory: speech model 0 = never, LLM -1 = never.
    asr_unload_minutes: int = 30
    llm_unload_minutes: int = 30
    mode_models: dict[str, str] = field(default_factory=dict)  # cleanup style -> its own LLM
    # Transcribe takes over 20 s in parts while they're still being recorded, so the text comes
    # sooner after the key's release.
    stream_long_takes: bool = True
    # Minutes the mic stays open after a take, so the next one starts recording at once instead
    # of after the 0.1-0.5 s it takes to open; 0 releases it after every take. Audio between
    # takes is dropped as it arrives, but the desktop's mic indicator stays on.
    mic_open_minutes: int = 0

    @classmethod
    def from_config(cls, raw: Mapping[str, Any], env: Mapping[str, str] = os.environ) -> "Settings":
        """The config file's raw values over the environment's, each checked: a bad or
        wrong-typed value (e.g. from a hand-edited file) falls back to its default."""
        s = {"sounds": env.get("ASR_SOUNDS", "1") != "0", "sound_volume": env.get("ASR_SOUND_VOLUME", "40"),
             "language": env.get("ASR_LANGUAGE") or "", "paste_key": env.get("ASR_PASTE_KEY", "shift+insert"),
             "rewrite_model": env.get("ASR_REWRITE_MODEL", "llama3.2"), "mode": "Clean up",
             "asr_model": env.get("ASR_MODEL", ""), "asr_unload_minutes": 30, "llm_unload_minutes": 30,
             "mode_models": {}, "stream_long_takes": env.get("ASR_STREAM", "1") != "0", "mic_open_minutes": 0}
        s.update({k: v for k, v in raw.items() if k in s and cls._fits(k, v)})
        try:
            s["sound_volume"] = min(max(int(s["sound_volume"]), 0), 100)
        except (TypeError, ValueError):
            s["sound_volume"] = 40
        if s["paste_key"] not in PASTE_KEYS:
            s["paste_key"] = "shift+insert"
        s["mode_models"] = {k: v for k, v in s["mode_models"].items() if isinstance(v, str)}
        for key, low in (("asr_unload_minutes", 0), ("llm_unload_minutes", -1)):
            if not isinstance(s[key], int) or isinstance(s[key], bool) or s[key] < low:
                s[key] = 30
        s["mic_open_minutes"] = max(s["mic_open_minutes"], 0)
        if not isinstance(s["stream_long_takes"], bool):
            s["stream_long_takes"] = True
        if s["language"] not in LANGUAGES:  # also catches a wrong type from a hand-edited config
            s["language"] = ""
        return cls(**s)

    @classmethod
    def _fits(cls, name: str, value: Any) -> bool:
        """Whether a config value has its setting's type (a bool isn't an int here)."""
        kind = cls.__dataclass_fields__[name].type
        if kind is int:
            return isinstance(value, int) and not isinstance(value, bool)
        return isinstance(value, dict if kind == dict[str, str] else kind)
