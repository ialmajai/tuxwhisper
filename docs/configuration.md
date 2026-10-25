# Configuration

## Settings menu

Open "TuxWhisper Settings" from your app grid (`./install.sh` adds it), run `tuxwhisper settings` (`./asr settings` in a git clone), or
bind `/home/<you>/tuxwhisper/asr settings` to a key, to change these without editing files:

- Sounds on or off, and their volume
- Language
- Paste key
- Keys: the key for each action. Pick from a list or type any combination, such as
  `Ctrl+Alt+D`; it takes effect at once. On GNOME only; other desktops set keys in their own
  settings ([Other desktops](other-desktops.md)).
- Cleanup style (Clean up, Email, Bullet points…), the cleanup model, and a model per style
- Speech model in memory and LLM in memory: how long each stays on the GPU when idle (default
  30 minutes), or always loaded. The speech model reloads in under a second from RAM, starting
  as you press the key, so you rarely notice.
- Microphone: released after each take (default), or kept open for a while after one: pick 5,
  10, 30 or 60 minutes, or type any time up to a day under Other….
  Kept open, recording starts the moment you press the key instead of after the 0.1-0.5 s the
  mic takes to open, so a word said straight away isn't clipped. Nothing is recorded between
  takes (the audio is dropped as it arrives), but your desktop's microphone indicator stays on
  until the mic is released.
- Speech model: Whisper models from tiny (fastest) to large-v3 (most accurate), with their
  download sizes. A downloaded model is used at once. A new one downloads in the background
  while you keep dictating with the current model, then TuxWhisper switches to it. If a model
  won't load, TuxWhisper goes back to the default.
- Vocabulary, replacements and custom cleanup styles
- Correct last transcript: fix what was pasted; Replace erases it and types the fix in its place
  (within 10 minutes of dictating, with the cursor still after it; otherwise the fix is copied
  for you to paste). Then tick what TuxWhisper should
  remember: a replacement rule (e.g. "tux whisper" → "TuxWhisper"), a vocabulary word, or the
  transcript of the recording you just saved with F4. Names and jargon come ticked; a fix of
  plain words (e.g. "two" → "three") doesn't, since as a rule it would change every "two". It opens
  with F6 by default; change the key under **Keys**.
- Recent transcripts: the last 20 this session; pick one to copy it again. TuxWhisper keeps
  them in `$XDG_RUNTIME_DIR`, which only you can read and which empties when you log out.

Most changes apply to your next dictation. The menu saves to `~/.config/tuxwhisper/config.toml`, which
overrides the environment variables below. It needs `zenity`, which GNOME-based distros
usually include; elsewhere, install the `zenity` package.

## Environment variables

Device, Whisper model, prompt and Ollama server are set only here; the menu's values take
priority for the rest. Add variables to the `[Service]` section of
`~/.config/systemd/user/tuxwhisper.service`, e.g. `Environment=ASR_DEVICE=cpu`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `ASR_LANGUAGE` | auto-detect | Language code, e.g. `en`. Setting it avoids misdetection on short clips. |
| `ASR_DEVICE` | `cuda` | `cuda` (NVIDIA), `vulkan` (AMD/Intel GPUs) or `cpu` |
| `ASR_MODEL` | `large-v3-turbo` (GPU), `small` (CPU) | Any faster-whisper model name or path; the settings menu's **Speech model** overrides it |
| `ASR_PROMPT` | a short punctuated sentence | Style example for Whisper; keeps capitals and punctuation. Used for English only. Set to empty to disable. |
| `ASR_SOUNDS` | `1` | `0` turns off the start and stop sounds |
| `ASR_STREAM` | `1` | `0` transcribes long takes only after the key's release, rather than in parts while you speak |
| `ASR_SOUND_VOLUME` | `40` | Sound volume, as a percent of the system volume |
| `ASR_PASTE_KEY` | `shift+insert` | `shift+insert`, `ctrl+v` or `ctrl+shift+v` |
| `ASR_REWRITE_MODEL` | `llama3.2` | Ollama model used for LLM cleanup |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server used for LLM cleanup |

Then apply the changes:

```bash
systemctl --user daemon-reload && systemctl --user restart tuxwhisper
```

TuxWhisper records from your default input device; change it in your desktop's sound
settings.

## Custom vocabulary

List names and jargon Whisper keeps misspelling in `~/.config/tuxwhisper/vocab.txt`, one per
line (`#` starts a comment). TuxWhisper re-reads the file on every dictation, so you don't need to restart.

```text
TuxWhisper
Ollama
Kubernetes
```

Keep it short: long lists can make Whisper insert the words where you didn't say them.
The vocabulary only applies to English dictation; in other languages it made Whisper
translate or garble the text.

## Replacements

List fixed rewrites in `~/.config/tuxwhisper/replacements.txt`, one `spoken => written` rule per
line. TuxWhisper applies them to the text before pasting, with or without LLM cleanup, matching whole words
in any case. `\n` inserts a line break, and a line-break rule also removes the punctuation
Whisper puts around it.

```text
new paragraph => \n\n
new line => \n
tux whisper => TuxWhisper
my email => you@example.com
```

Like `vocab.txt`, the file is re-read on every dictation. Saved recordings keep Whisper's original
text, which matches the audio. Pick phrases you won't say for real: a `comma => ,` rule also
fires when you mean the word.

## AMD and Intel GPUs (Vulkan)

On AMD and Intel GPUs, TuxWhisper runs Whisper through [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
and Vulkan instead of faster-whisper and CUDA. `./install.sh` picks this when it finds an AMD or
Intel GPU and no NVIDIA one. It compiles whisper.cpp on install, which takes a few minutes and needs
a C++ compiler, CMake, the Vulkan headers and shader tools; the installer adds those packages.
Force it with `./install.sh --engine=vulkan`, or go back with `--engine=cpu`.

- **Untested on AMD and Intel hardware.** It's tested through Vulkan on an NVIDIA RTX 3080, where
  a 7 s clip takes about 0.4 s (CUDA: 0.25 s). Integrated Intel graphics may be slower than the
  CPU; if so, use `--engine=cpu`. Reports are welcome.
- **Vocabulary is weaker:** whisper.cpp has no hotwords, so `vocab.txt` goes into Whisper's prompt.
- **A separate model download** (about 1.6 GB) to `~/.local/share/pywhispercpp/models`, plus
  a 1 MB voice activity model that skips silence, so pauses don't turn into invented text.

## CPU mode

`./install.sh` sets this up when it finds no supported GPU. By hand, use the **No NVIDIA GPU (CPU
mode)** block in step 1 of the [manual install](manual-install.md#1-clone-and-install): it
installs the CPU build and sets `ASR_DEVICE=cpu` in a drop-in file. To switch an existing
install, run those commands, then
`systemctl --user daemon-reload && systemctl --user restart tuxwhisper`.
TuxWhisper then uses the smaller `small` model: a 3–4 s clip takes about 1.4 s on a recent
desktop CPU.

## Paste key

The default, Shift+Insert, pastes in browsers, editors and terminals on any keyboard
layout. If an app doesn't paste, try `ctrl+v`, which works in most GUI apps but
not in terminals.
