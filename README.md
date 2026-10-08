# 🐧 TuxWhisper

**Offline, hotkey-driven dictation for Linux.** Press a key, speak, and your words are
typed wherever your cursor is: a browser text box, an LLM chat prompt, your editor or
your terminal.

[![CI](https://github.com/ialmajai/tuxwhisper/actions/workflows/ci.yml/badge.svg)](https://github.com/ialmajai/tuxwhisper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-blue?logo=linux&logoColor=white)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![X11 | Wayland](https://img.shields.io/badge/display-X11%20%7C%20Wayland-555)

Speech recognition runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper);
your audio never leaves your machine.

## Features

- **Types anywhere:** browsers, editors, terminals, on X11 and Wayland (including GNOME).
- **Fast:** about half a second from releasing the key to text on screen, with an NVIDIA GPU.
- **Tap or hold:** tap to start and stop, or hold to talk.
- **LLM cleanup (optional):** a local [Ollama](https://ollama.com) model removes "um"s and
  fixes punctuation.
- **Voice dataset:** save takes with their transcripts for fine-tuning.
- **Clipboard-safe:** TuxWhisper restores whatever you had copied.

Tested on GNOME (Wayland) with an NVIDIA GPU, and in CPU mode.

## Usage

| Key | Action |
|-----|--------|
| **F5** | Tap to start, tap to stop, or hold to talk. The text is pasted at your cursor. |
| **F3** | Same as F5, with [LLM cleanup](docs/llm-cleanup.md). |
| **F4** | [Save the last take](docs/saving-takes.md) for fine-tuning. |

Wait for the "🎙 Recording…" notification before speaking.

## Quick start

**Requirements:** Linux with systemd, Python 3.12, [uv](https://docs.astral.sh/uv/), an
NVIDIA GPU with about 1.5 GB of free VRAM (or [CPU mode](docs/configuration.md#cpu-mode)), and `xclip`
(or `wl-clipboard` on Wayland without XWayland).

**1. Clone and install**

```bash
git clone https://github.com/ialmajai/tuxwhisper.git ~/tuxwhisper
cd ~/tuxwhisper
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements-cuda.txt   # no NVIDIA GPU: requirements.txt
```

**2. Allow the virtual keyboard** (needs sudo once; read the [security note](docs/security.md))

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

Use the full path; shortcut commands don't expand `~`. KDE, Hyprland, Sway, i3: see
[Other desktops](docs/other-desktops.md).

**5. Optional: LLM cleanup.** Install Ollama ([Linux install guide](https://docs.ollama.com/linux)) and the cleanup model:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
```

Click into any text box, press F5 and speak.

> [!TIP]
> Binding F5 overrides page refresh in browsers. Ctrl+R still refreshes.

## Documentation

- [LLM cleanup (F3)](docs/llm-cleanup.md)
- [Saving takes for fine-tuning (F4)](docs/saving-takes.md)
- [Configuration](docs/configuration.md): language, model, custom vocabulary, replacements, CPU mode, paste key
- [Other desktops](docs/other-desktops.md)
- [How it works](docs/how-it-works.md)
- [Security](docs/security.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

[MIT](LICENSE) © 2026 Ibrahim Almajai
