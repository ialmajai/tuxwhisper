# How it works

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
  (`/dev/uinput`) presses the paste key. About 0.3 s later, TuxWhisper restores your
  previous clipboard (text only; you lose a copied image).
- **Hold detection:** while a key is held, the desktop re-runs the shortcut on auto-repeat.
  A burst of presses counts as a held key, and recording stops when the repeats end.

## Commands

The keys run small commands, which you can also use from a terminal or script:

```bash
./asr toggle    # start / stop dictation
./asr rewrite   # start / stop dictation with LLM cleanup
./asr save      # save the last take
./asr serve     # run the daemon in the foreground (normally systemd does this)
```

TuxWhisper ignores clips under 0.3 s; Whisper invents text on near-silence.
