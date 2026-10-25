# How it works

```text
 keys ──────────▶ asr toggle|rewrite|save ──▶ Unix socket ──▶ asr serve (daemon)
                                                                 │
     mic ──▶ record ──▶ faster-whisper ──▶ (Ollama cleanup) ─────┤
                                                                 ▼
        your app ◀── Shift+Insert (virtual keyboard) ◀── clipboard
```

- **Daemon:** `asr serve` keeps the Whisper model loaded and listens on a Unix socket in
  `$XDG_RUNTIME_DIR`, which only your user can access.
- **Key presses:** each press runs `asr toggle` (or `rewrite`, `save`), which only passes the
  word to the daemon. In a git clone with [Rust](https://rustup.rs) installed, `install.sh`
  builds a small client for this in `client-rs/`: it takes about 1 ms where starting Python
  takes about 80. Without it, or when the daemon isn't running, the Python client does it.
- **Typing:** Wayland doesn't let apps type into other windows, so TuxWhisper puts the
  text on the clipboard (`xclip`, or `wl-clipboard` without XWayland) and presses the paste
  key through a virtual keyboard (`/dev/uinput`). Shortly after, it restores your previous
  clipboard, images included (rich text comes back as plain text).
- **Hold detection:** while you hold a key, the desktop re-runs the shortcut on auto-repeat.
  TuxWhisper treats a burst of presses as a held key and stops recording when the repeats end.

## Commands

The keys run small commands, which you can also use from a terminal or script. From PyPI the
command is `tuxwhisper`; in a git clone, `./asr` runs the same commands:

```bash
tuxwhisper toggle     # start / stop dictation
tuxwhisper rewrite    # start / stop dictation with LLM cleanup
tuxwhisper save       # save the last recording
tuxwhisper settings   # open the settings menu
tuxwhisper bind       # bind the default keys on GNOME
tuxwhisper correct    # fix the last transcript and learn from it
tuxwhisper serve      # run the daemon in the foreground (systemd does this at login)
tuxwhisper setup      # install packages, service, shortcuts (same as ./install.sh)
tuxwhisper uninstall  # remove what setup added (same as ./uninstall.sh)
```

TuxWhisper ignores clips under 0.3 s; Whisper invents text on near-silence.
