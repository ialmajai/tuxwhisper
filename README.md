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
  types your words about half a second after you release the key, with an NVIDIA GPU. AMD
  and Intel GPUs work through Vulkan.
- **Types anywhere:** browsers, editors, terminals, on X11 and Wayland (including GNOME).
- **LLM cleanup (optional):** the cleanup key runs your words through a local [Ollama](https://ollama.com)
  model (Meta's Llama 3.2 by default) that removes "um"s and fixes punctuation, or rewrites
  them as an email, bullet points or your own style. The model must fit in the GPU memory of
  the machine running Ollama; otherwise it pastes the raw transcript.
- **Many languages:** dictate in any of Whisper's 100 languages. LLM cleanup replies in the language
  you spoke; Llama 3.2 is weak outside English, so [pick a stronger model](docs/llm-cleanup.md)
  for other languages.

**Also:** tap or hold to talk, a custom vocabulary for names and jargon, replacements such as
"new line" → line break, corrections that teach it your words for next time, a settings menu
with your last 20 transcripts, and
[saved recordings](docs/saving-takes.md) for fine-tuning Whisper to your voice and accent.
TuxWhisper types by pasting, then puts back the text you had copied.

Tested on GNOME (Wayland) with an NVIDIA GPU and in CPU mode, and on
[Sway 1.12 and i3 4.25](docs/other-desktops.md) in containers.

## Usage

| Key | Action |
|-----|--------|
| **Dictation key** | Tap to start, tap to stop, or hold to talk. The text is typed at your cursor. |
| **Cleanup key** | The same, with [LLM cleanup](docs/llm-cleanup.md). |
| **Save key** | [Save the last recording](docs/saving-takes.md) for fine-tuning, after you check its transcript. |
| **Correct key** | Fix the last transcript: the fix replaces it, and TuxWhisper can remember it for next time. |
| **Settings key** | Open the [settings menu](docs/configuration.md#settings-menu). |

On GNOME, setup binds them to F5, F3, F4, F6 and Alt+F9; change any of them under **Keys** in the
settings menu.
Wait for the "🎙 Recording…" notification and the start sound before speaking.

## Install

You need Linux with systemd, [uv](https://docs.astral.sh/uv/), and a GPU with about 1.5 GB of
free memory: NVIDIA (CUDA), or AMD or Intel ([Vulkan](docs/configuration.md#amd-and-intel-gpus-vulkan),
not yet tested on those). No GPU? TuxWhisper runs on the CPU too, more slowly.
No uv yet? Its official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install TuxWhisper from [PyPI](https://pypi.org/project/tuxwhisper/) and set it up:

```bash
uv tool install tuxwhisper        # NVIDIA GPU: uv tool install 'tuxwhisper[cuda]'
tuxwhisper setup
```

Setup says what each step does and asks before anything that needs your password. It installs
the system packages on Debian, Ubuntu, Fedora, Arch and openSUSE, the background service, and on
GNOME the keys. It's safe to run again; [read it](install.sh) first if you like. On other distros,
use the manual install below.

Prefer a git clone? `git clone https://github.com/ialmajai/tuxwhisper.git && cd tuxwhisper &&
./install.sh` does the same.

**Advanced:** [Manual install](docs/manual-install.md) runs every step by hand.
[Other desktops](docs/other-desktops.md) covers KDE, Hyprland, Sway and i3.

### Try it

Click into any text box, hold the dictation key, wait for the start sound, speak, and release. Nothing
typed? See [Troubleshooting](docs/troubleshooting.md).

> [!TIP]
> The default dictation, cleanup and correct keys, F5, F3 and F6, replace page refresh,
> find-next and jump-to-address-bar in browsers and many editors. Ctrl+R, Ctrl+G and Ctrl+L
> still do those, or pick other keys under **Keys**.

### Optional: LLM cleanup

The cleanup key needs [Ollama](https://ollama.com) and about 2 GB more GPU memory:
[set it up](docs/llm-cleanup.md#install).

## Uninstall

```bash
tuxwhisper uninstall    # in a git clone: ./uninstall.sh
```

It asks before removing the virtual keyboard rule, and keeps your settings, saved recordings and
downloaded models, listing where they are.

## Documentation

- [Manual install](docs/manual-install.md)
- [LLM cleanup](docs/llm-cleanup.md)
- [Saving recordings for fine-tuning](docs/saving-takes.md)
- [Configuration](docs/configuration.md): settings menu, custom vocabulary, replacements, GPU and CPU modes
- [Other desktops](docs/other-desktops.md)
- [How it works](docs/how-it-works.md)
- [Security](docs/security.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

[MIT](LICENSE) © 2026 Ibrahim Almajai. The penguin icon is from
[Noto Emoji](https://github.com/googlefonts/noto-emoji) (Apache License 2.0).
