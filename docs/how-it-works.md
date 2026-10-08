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
- **Typing:** Wayland doesn't let apps type into other windows, so TuxWhisper puts the
  text on the clipboard (`xclip`, or `wl-clipboard` without XWayland) and presses the paste
  key through a virtual keyboard (`/dev/uinput`). About 0.3 s later, TuxWhisper restores your
  previous clipboard (text only; you lose a copied image).
- **Hold detection:** while you hold a key, the desktop re-runs the shortcut on auto-repeat.
  TuxWhisper treats a burst of presses as a held key and stops recording when the repeats end.

## Commands

The keys run small commands, which you can also use from a terminal or script:

```bash
./asr toggle    # start / stop dictation
./asr rewrite   # start / stop dictation with LLM cleanup
./asr save      # save the last take
./asr serve     # run the daemon in the foreground (systemd does this at login)
```

TuxWhisper ignores clips under 0.3 s; Whisper invents text on near-silence.
