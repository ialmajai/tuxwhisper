# Configuration

## Settings menu

Open "TuxWhisper Settings" from your app grid (see the Quick start), run `./asr settings`, or
bind `/home/<you>/tuxwhisper/asr settings` to a key, to change these without editing files:

- Sounds on or off, and their volume
- Language
- Paste key
- F3 mode (Clean up, Email, Bullet points…) and LLM model
- Vocabulary, replacements and custom F3 modes
- Recent transcripts: the last 20 this session; pick one to copy it again. TuxWhisper keeps
  them in `$XDG_RUNTIME_DIR`, which only you can read and which empties when you log out.

Changes apply on the next take. The menu saves to `~/.config/tuxwhisper/config.toml`, which
overrides the environment variables below. It needs `zenity`, which GNOME-based distros
usually include; elsewhere, install the `zenity` package.

## Environment variables

Device, Whisper model, prompt and Ollama server are set only here; the menu's values take
priority for the rest. Add variables to the `[Service]` section of
`~/.config/systemd/user/tuxwhisper.service`, e.g. `Environment=ASR_DEVICE=cpu`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `ASR_LANGUAGE` | auto-detect | Language code, e.g. `en`. Setting it avoids misdetection on short clips. |
| `ASR_DEVICE` | `cuda` | `cuda` or `cpu` |
| `ASR_MODEL` | `large-v3-turbo` (GPU), `small` (CPU) | Any faster-whisper model name or path |
| `ASR_PROMPT` | a short punctuated sentence | Style example for Whisper; keeps capitals and punctuation. Set to empty to disable. |
| `ASR_SOUNDS` | `1` | `0` turns off the start and stop sounds |
| `ASR_SOUND_VOLUME` | `40` | Sound volume, as a percent of the system volume |
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

## Replacements

List fixed rewrites in `replacements.txt` next to `asr.py`, one `spoken => written` rule per
line. TuxWhisper applies them to the text before pasting, on F5 and F3, matching whole words
in any case. `\n` inserts a line break, and a line-break rule also removes the punctuation
Whisper puts around it.

```text
new paragraph => \n\n
new line => \n
tux whisper => TuxWhisper
my email => you@example.com
```

Like `vocab.txt`, the file is re-read on every take. Saved takes (F4) keep Whisper's original
text, which matches the audio. Pick phrases you won't say for real: a `comma => ,` rule also
fires when you mean the word.

## CPU mode

Without an NVIDIA GPU, install from `requirements.txt` and set `Environment=ASR_DEVICE=cpu`.
TuxWhisper then uses the smaller `small` model: a 3–4 s clip takes about 1.4 s on a recent
desktop CPU.

## Paste key

The default, Shift+Insert, pastes in browsers, editors and terminals on any keyboard
layout. If an app doesn't paste, try `ctrl+v`, which works in most GUI apps but
not in terminals.
