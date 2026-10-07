# 🐧 TuxWhisper

**Offline, hotkey-driven dictation for Linux.** Press a key, speak, and your words are
typed wherever your cursor is: a browser text box, an LLM chat prompt, your editor or
your terminal.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-blue?logo=linux&logoColor=white)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![X11 | Wayland](https://img.shields.io/badge/display-X11%20%7C%20Wayland-555)

Speech recognition runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper);
nothing is sent to the cloud.

## Features

- **Types anywhere.** Works in any app that accepts paste, on X11 and Wayland, including
  GNOME on Wayland, where most typing tools don't work.
- **Fast.** About half a second from releasing the key to text on screen, with an NVIDIA GPU.
- **Tap or hold.** Tap to start and stop, or hold the key to talk and release it to transcribe.
- **Optional LLM cleanup.** A second key passes the transcript through a local
  [Ollama](https://ollama.com) model to fix punctuation and remove "um"s, without answering
  the questions you dictate.
- **Builds a voice dataset.** One key saves the last recording with its transcript, in a
  format that loads straight into Hugging Face `datasets`.
- **Leaves your clipboard alone.** Whatever you had copied is restored after each paste.
- **Works on any keyboard layout.** It pastes with Shift+Insert, which doesn't depend on
  QWERTY.

Developed and tested on GNOME (Wayland) with an NVIDIA GPU, and in CPU mode. Other
desktops should work but are untested; see [Other desktops](#other-desktops).

## Usage

| Key | Action |
|-----|--------|
| **F5** (tap) | Start recording. Tap again to stop, transcribe and paste. |
| **F5** (hold) | Push-to-talk: speak while holding, release to transcribe and paste. |
| **F3** | Same as F5, but the transcript is cleaned up by a local LLM before pasting. |
| **F4** | Save the last take (audio + transcript) to `recordings/`. |

The keys are only suggestions; you choose them when you set up the shortcuts.

- **Wait for the "🎙 Listening…" notification before speaking.** The microphone takes about
  0.3 s to open, and anything said before that is lost.
- **Very short clips are ignored.** Anything under 0.3 s produces no text.

The keys run small commands, which you can also use from a terminal or script:

```bash
./asr toggle    # start / stop dictation
./asr rewrite   # start / stop dictation with LLM cleanup
./asr save      # save the last take
./asr serve     # run the daemon in the foreground (normally systemd does this)
```

## Quick start

**Requirements:** Linux with systemd, Python 3.12, [uv](https://docs.astral.sh/uv/), an
NVIDIA GPU with about 1.5 GB of free VRAM (or see [CPU mode](#cpu-mode)), and `xclip`
(or `wl-clipboard` on Wayland without XWayland).

**1. Clone and install**

```bash
git clone https://github.com/ialmajai/tuxwhisper.git ~/tuxwhisper
cd ~/tuxwhisper
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-cuda.txt   # no NVIDIA GPU: requirements.txt
```

**2. Allow the virtual keyboard** (needs sudo once; read the [security note](#security-note))

```bash
echo 'KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/60-uinput.rules
sudo udevadm control --reload && sudo udevadm trigger --name-match=uinput
```

**3. Start the daemon at login**

```bash
cp tuxwhisper.service ~/.config/systemd/user/
systemctl --user enable --now tuxwhisper
```

The service expects the repo at `~/tuxwhisper`. If it's elsewhere, edit `ExecStart` in the
copied file. The first start downloads the Whisper model (about 1.5 GB) to
`~/.cache/huggingface`, so it takes a while; after that, startup takes about 15 s.

**4. Bind the keys.** On GNOME: Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts.

| Shortcut | Command |
|---|---|
| F5 | `/home/<you>/tuxwhisper/asr toggle` |
| F3 | `/home/<you>/tuxwhisper/asr rewrite` |
| F4 | `/home/<you>/tuxwhisper/asr save` |

Use the full path; shortcut commands don't expand `~`. For other desktops, see
[Other desktops](#other-desktops).

**5. Optional: LLM cleanup.** Install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2
```

Click into any text box, press F5 and speak.

> [!TIP]
> Binding F5 overrides page refresh in browsers. Ctrl+R still refreshes.

## LLM cleanup (F3)

F3 sends the transcript to a local Ollama model (`llama3.2` by default). The model fixes
punctuation and capitalization and removes filler words ("um", "uh", "like"), repeated words
and false starts.

```text
Whisper:  um so can you like uh explain how the the attention mechanism works
Pasted:   So can you explain how the attention mechanism works?
```

- **It cleans; it doesn't answer.** A dictated question is pasted as a question, not
  answered.
- **Speed:** about 0.2–1 s once the model is loaded. The model unloads after 30 minutes
  idle to free GPU memory, so the next F3 takes a few seconds longer.
- **Always pastes something:** if Ollama is unreachable, or the model doesn't fit in free
  GPU memory, the raw transcript is pasted and a notification says why.
- **Not perfect:** it sometimes drops real words. Use F5 when the exact wording matters.

Ollama can run on another machine; point `OLLAMA_URL` at it (see
[Configuration](#configuration)).

## Saving takes (F4)

F4 saves the most recent take to `recordings/`:

```text
recordings/
├── 20260101-120000.wav    # 16 kHz, mono, 16-bit
└── metadata.csv           # file_name,transcription
```

This is the Hugging Face `audiofolder` layout, so the folder loads directly:

```python
from datasets import load_dataset
ds = load_dataset("audiofolder", data_dir="recordings")
```

- Only the **latest** take can be saved, and only once. Pressing F4 again shows "Nothing to
  save" until you dictate again.
- The saved transcript is Whisper's **raw** output, even for F3 takes, because that's what
  matches the audio. Corrections you make in the text box aren't saved; edit `metadata.csv`
  if needed.

## Configuration

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

Audio comes from your default input device, which you can change in your desktop's sound
settings.

### CPU mode

Without an NVIDIA GPU, install from `requirements.txt` and set `Environment=ASR_DEVICE=cpu`.
TuxWhisper then uses the smaller `small` model: a 3–4 s clip takes about 1.4 s on a recent
desktop CPU.

### Paste key

The default, Shift+Insert, pastes in browsers, editors and terminals on any keyboard
layout. If a particular app doesn't paste, try `ctrl+v`, which works in most GUI apps but
not in terminals.

## Other desktops

Bind the same commands in your desktop's shortcut settings:

| Desktop | Binding | Hold to talk |
|---|---|---|
| GNOME | Settings → Keyboard → Custom Shortcuts | ✅ Yes |
| KDE Plasma | System Settings → Shortcuts → Add New → Command | ❔ Untested |
| Hyprland | `binde = , F5, exec, ~/tuxwhisper/asr toggle` | ✅ Should work (`binde` repeats while held) |
| Sway | `bindsym F5 exec ~/tuxwhisper/asr toggle` | ❌ Tap only |
| i3 | `bindsym F5 exec --no-startup-id ~/tuxwhisper/asr toggle` | ❌ Tap only |

Where holding doesn't work, tap to start and tap again to stop.

The systemd service starts with `graphical-session.target`, which some window managers (i3,
or Sway without extra setup) never start. On those, skip step 3 and start the daemon from
your WM config instead, e.g. `exec ~/tuxwhisper/asr serve`.

## How it works

```text
 F5 / F3 / F4 ──▶ asr toggle|rewrite|save ──▶ Unix socket ──▶ asr serve (daemon)
                                                                 │
     mic ──▶ record ──▶ faster-whisper ──▶ (Ollama cleanup) ─────┤
                                                                 ▼
        your app ◀── Shift+Insert (virtual keyboard) ◀── clipboard
```

- **Daemon:** `asr serve` keeps the Whisper model loaded and listens on a Unix socket in
  `$XDG_RUNTIME_DIR`, which only your user can access.
- **Typing:** Wayland doesn't let apps type into other windows, so the text goes on the
  clipboard (`xclip`, or `wl-clipboard` without XWayland) and a virtual keyboard
  (`/dev/uinput`) presses the paste key. The previous clipboard is restored about 0.3 s
  later (text only; a copied image is lost).
- **Hold detection:** while a key is held, the desktop re-runs the shortcut on auto-repeat.
  A burst of presses counts as a held key, and recording stops when the repeats end.

## Security note

> [!WARNING]
> Step 2 gives your user write access to `/dev/uinput`. TuxWhisper needs it to press the
> paste key, but it also lets **any program you run create a virtual keyboard or mouse and
> send input to any window**, including terminals and password prompts. Tools like
> `ydotool` require the same access.

The `uaccess` tag limits this to the user logged in at the active local session, and only
while that session is active. Other user accounts don't get access, but every process
running as you does, including ones started over SSH while you're logged in.

If you don't want this, skip step 2: transcription still works, but the text isn't pasted.
To undo it later, run `sudo rm /etc/udev/rules.d/60-uinput.rules` and reboot.

## Troubleshooting

```bash
systemctl --user status tuxwhisper    # is the daemon running?
journalctl --user -u tuxwhisper -f    # live log: each transcription, timings and errors
systemctl --user restart tuxwhisper   # restart (the model takes ~15 s to load)
```

<details>
<summary><b>Nothing happens when I press the key</b></summary>

Check `systemctl --user status tuxwhisper`. Right after login, the model may still be loading
(watch for `ready` in the log). If you see "daemon is not running", start it with
`systemctl --user start tuxwhisper`. Also check that the shortcut command uses the full path.
</details>

<details>
<summary><b>It transcribes but nothing is pasted</b></summary>

Run `getfacl /dev/uinput`; it should list `user:<you>:rw-`. If not, check the udev rule from
step 2, including the file name (it must start with a number below 73).
</details>

<details>
<summary><b>It doesn't paste in one particular app</b></summary>

That app may not treat Shift+Insert as paste. Try `ASR_PASTE_KEY=ctrl+v`.
</details>

<details>
<summary><b>My old clipboard is pasted instead of the transcript</b></summary>

The app read the clipboard after it had already been restored. Increase the `0.3` s delay in
`paste()` in `asr.py`.
</details>

<details>
<summary><b>The daemon exits with "cuda unavailable"</b></summary>

The GPU or the CUDA libraries couldn't be used. Check that you installed from
`requirements-cuda.txt` and that `nvidia-smi` works, or switch to [CPU mode](#cpu-mode).
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

Another program is using the GPU. Free some memory, or run Ollama for F3 on another machine
with `OLLAMA_URL`.
</details>

<details>
<summary><b>F3 pastes the raw transcript</b></summary>

The notification says why: either Ollama isn't reachable (check `ollama list` and
`OLLAMA_URL`), or there isn't enough free GPU memory for the cleanup model.
</details>

## Project layout

```text
asr                    launcher wrapper (use this, not asr.py directly)
asr.py                 daemon and client commands
tuxwhisper.service     systemd user service template
requirements.txt       Python dependencies
requirements-cuda.txt  + NVIDIA CUDA libraries
recordings/            saved takes (created by F4, not committed)
```

## License

[MIT](LICENSE) © 2026 Ibrahim Almajai
