# Configuration

Settings are environment variables. Add them to the `[Service]` section of
`~/.config/systemd/user/tuxwhisper.service`, e.g. `Environment=ASR_LANGUAGE=en`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `ASR_LANGUAGE` | auto-detect | Language code, e.g. `en`. Setting it avoids misdetection on short clips. |
| `ASR_DEVICE` | `cuda` | `cuda` or `cpu` |
| `ASR_MODEL` | `large-v3-turbo` (GPU), `small` (CPU) | Any faster-whisper model name or path |
| `ASR_PROMPT` | a short punctuated sentence | Style example for Whisper; keeps capitals and punctuation. Set to empty to disable. |
| `ASR_PASTE_KEY` | `shift+insert` | `shift+insert`, `ctrl+v` or `ctrl+shift+v` |
| `ASR_REWRITE_MODEL` | `llama3.2` | Ollama model used by F3 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server used by F3 |

Then apply the changes:

```bash
systemctl --user daemon-reload && systemctl --user restart tuxwhisper
```

TuxWhisper records from your default input device; change it in your desktop's sound
settings.

## Custom vocabulary

List names and jargon Whisper keeps misspelling in `vocab.txt` next to `asr.py`, one per
line (`#` starts a comment). TuxWhisper re-reads the file on every take, so you don't need to restart.

```text
TuxWhisper
Ollama
Kubernetes
```

Keep it short: long lists can make Whisper insert the words where you didn't say them.

## CPU mode

Without an NVIDIA GPU, install from `requirements.txt` and set `Environment=ASR_DEVICE=cpu`.
TuxWhisper then uses the smaller `small` model: a 3–4 s clip takes about 1.4 s on a recent
desktop CPU.

## Paste key

The default, Shift+Insert, pastes in browsers, editors and terminals on any keyboard
layout. If an app doesn't paste, try `ctrl+v`, which works in most GUI apps but
not in terminals.
