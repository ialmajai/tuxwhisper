# 🐧 TuxWhisper

**Private, offline dictation for Linux.** Press a key, speak, and your words are typed
wherever your cursor is: a browser text box, an LLM chat prompt, your editor or your
terminal. Everything runs on your own machine with free, open models. Your voice never
reaches a cloud service, and there's nothing to pay for.

[![CI](https://github.com/ialmajai/tuxwhisper/actions/workflows/ci.yml/badge.svg)](https://github.com/ialmajai/tuxwhisper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform: Linux](https://img.shields.io/badge/platform-Linux-blue?logo=linux&logoColor=white)
![X11 | Wayland](https://img.shields.io/badge/display-X11%20%7C%20Wayland-555)

![Animated re-enactment of TuxWhisper: tap the dictation key, speak, tap again, and the sentence is typed into Google's search box in Chrome; in a terminal running Claude Code, the cleanup key also removes filler words; the settings key opens the settings menu](docs/demo.gif)

<sub>Re-enactment over real screenshots; the cleaned-up sentence is a local LLM's actual output.</sub>

## Features

- **Fast:** OpenAI's open Whisper model ([faster-whisper](https://github.com/SYSTRAN/faster-whisper))
  types your words about half a second after you release the key, with an NVIDIA GPU.
- **Types anywhere:** browsers, editors, terminals, on X11 and Wayland (including GNOME).
- **LLM cleanup (optional):** F3 runs your words through a local [Ollama](https://ollama.com)
  model (Meta's Llama 3.2 by default) that removes "um"s and fixes punctuation, or rewrites
  them as an email, bullet points or your own mode. The model must fit in the GPU memory of
  the machine running Ollama; otherwise F3 pastes the raw transcript.
- **Many languages:** dictate in any of Whisper's 100 languages. F3 replies in the language
  you spoke; Llama 3.2 is weak outside English, so [pick a stronger model](docs/llm-cleanup.md)
  for other languages.

**Also:** tap or hold to talk, a custom vocabulary for names and jargon, replacements such as
"new line" → line break, a settings menu with your last 20 transcripts, and
[saved recordings](docs/saving-takes.md) for fine-tuning Whisper to your voice and accent.
TuxWhisper types by pasting, then puts back the text you had copied.

Tested on GNOME (Wayland) with an NVIDIA GPU and in CPU mode, and on
[Sway 1.12 and i3 4.25](docs/other-desktops.md) in containers.

## Usage

| Key | Action |
|-----|--------|
| **F5** | Tap to start, tap to stop, or hold to talk. The text is typed at your cursor. |
| **F3** | Same as F5, with [LLM cleanup](docs/llm-cleanup.md). |
| **F4** | [Save the last recording](docs/saving-takes.md) and its transcript for fine-tuning. |
| **Alt+F9** | Open the [settings menu](docs/configuration.md#settings-menu). |

The installer binds these keys on GNOME. Wait for the "🎙 Recording…" notification and the start
sound before speaking.

## Install

You need Linux with systemd, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU with about
1.5 GB of free GPU memory. No NVIDIA GPU? TuxWhisper runs on the CPU too, more slowly.
No uv yet? Its official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/ialmajai/tuxwhisper.git ~/tuxwhisper
cd ~/tuxwhisper
./install.sh
```

The installer says what each step does and asks before anything that needs your password.
It's safe to run again. [Read it](install.sh) first if you like. It installs system packages
on Debian, Ubuntu, Fedora, Arch and openSUSE; on other distros, use the manual install below.

**Advanced:** [Manual install](docs/manual-install.md) runs every step by hand.
[Other desktops](docs/other-desktops.md) covers KDE, Hyprland, Sway and i3.

### Try it

Click into any text box, hold F5, wait for the start sound, speak, and release. Nothing
typed? See [Troubleshooting](docs/troubleshooting.md).

> [!TIP]
> F5 and F3 replace page refresh and find-next in browsers and many editors. Ctrl+R and Ctrl+G
> still do those.

### Optional: LLM cleanup (F3)

F3 needs [Ollama](https://ollama.com) and about 2 GB more GPU memory:
[set it up](docs/llm-cleanup.md#install).

## Uninstall

```bash
./uninstall.sh
```

It asks before removing the virtual keyboard rule, and keeps your settings, saved recordings and
downloaded models, listing where they are.

## Documentation

- [Manual install](docs/manual-install.md)
- [LLM cleanup (F3)](docs/llm-cleanup.md)
- [Saving recordings for fine-tuning (F4)](docs/saving-takes.md)
- [Configuration](docs/configuration.md): settings menu, custom vocabulary, replacements, CPU mode
- [Other desktops](docs/other-desktops.md)
- [How it works](docs/how-it-works.md)
- [Security](docs/security.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

[MIT](LICENSE) © 2026 Ibrahim Almajai
