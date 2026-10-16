# TuxWhisper

**Private, offline dictation for Linux.** Press a key, speak, and your words are typed wherever
your cursor is: a browser text box, an LLM chat prompt, your editor or your terminal. Speech
recognition ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)) and the optional LLM
cleanup ([Ollama](https://ollama.com)) run on your own machine with free, open models.

![TuxWhisper demo](https://raw.githubusercontent.com/ialmajai/tuxwhisper/main/docs/demo.gif)

## Install

```bash
uv tool install tuxwhisper        # NVIDIA GPU: uv tool install 'tuxwhisper[cuda]'
tuxwhisper setup
```

`tuxwhisper setup` installs the system packages it needs (Debian, Ubuntu, Fedora, Arch and
openSUSE), sets up the background service and, on GNOME, binds the keys. It says what each step
does and asks before anything that needs your password. On an AMD or Intel GPU, it builds the
Vulkan engine for you.

Then click into any text box, press the dictation key, wait for the start sound, speak, and press
it again.

| Key | Action |
|-----|--------|
| **Dictation key** | Tap to start and stop, or hold to talk |
| **Cleanup key** | The same, with local LLM cleanup |
| **Save key** | Check and save the last recording for fine-tuning |
| **Correct key** | Fix the last transcript in place |
| **Settings key** | Settings menu |

On GNOME, setup binds them to F5, F3, F4, F6 and Alt+F9; change them under **Keys** in the settings
menu.

Documentation, other desktops and troubleshooting: <https://github.com/ialmajai/tuxwhisper>
