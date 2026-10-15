# Troubleshooting

```bash
systemctl --user status tuxwhisper    # is the daemon running?
journalctl --user -u tuxwhisper -f    # live log: each transcription, timings and errors
systemctl --user restart tuxwhisper   # restart (the model takes ~15 s to load)
```

<details>
<summary><b>"Your mic is clipping"</b></summary>

The recording was louder than the microphone can capture, so peaks were cut off, which makes
transcripts less accurate. TuxWhisper still pastes the text, and warns at most every 10 minutes.
Move the mic a little further from your mouth, or lower its input volume in Settings → Sound.
</details>

<details>
<summary><b>Nothing happens when I press the key</b></summary>

Check `systemctl --user status tuxwhisper`. Right after login, the model may still be loading
(watch for `ready` in the log). If you see "daemon is not running", start it with
`systemctl --user start tuxwhisper`. Also check that the shortcut command uses the full path.
</details>

<details>
<summary><b>"No virtual keyboard: transcripts are copied"</b></summary>

TuxWhisper couldn't open `/dev/uinput`, so it copies each transcript for you to paste with
Ctrl+V. To have it type again, run `getfacl /dev/uinput`; it should list `user:<you>:rw-`. If
not, run `./install.sh` again and allow the virtual keyboard, or check the rule from step 2 of
the [manual install](manual-install.md#2-allow-the-virtual-keyboard), including the file name
(it must start with a number below 73). Then restart the daemon.
</details>

<details>
<summary><b>It doesn't paste in one app</b></summary>

That app may not treat Shift+Insert as paste. Set **Paste key** to `ctrl+v` in the settings
menu.
</details>

<details>
<summary><b>It pastes my old clipboard instead of the transcript</b></summary>

The app read the clipboard after TuxWhisper had restored it. Increase the `0.3` s delay in
`paste()` in `tuxwhisper/asr.py`.
</details>

<details>
<summary><b>The daemon exits with "cuda unavailable"</b></summary>

TuxWhisper couldn't use the GPU or the CUDA libraries. Check that `nvidia-smi` works and run
`./install.sh` (or `tuxwhisper setup`) again, which adds the CUDA libraries, or switch to
[CPU mode](configuration.md#cpu-mode).
</details>

<details>
<summary><b>"Not enough GPU memory" for the speech model</b></summary>

Another program is using the GPU (`nvidia-smi` shows which). TuxWhisper then uses the largest
downloaded speech model that fits, says so, and goes back to yours once there's room again. If
not even the smallest fits, it tells you and tries again each time you dictate. Free some
memory, set `ASR_DEVICE=cpu` (see [CPU mode](configuration.md#cpu-mode)), or run Ollama for
LLM cleanup on another machine with `OLLAMA_URL`.
</details>

<details>
<summary><b>LLM cleanup pastes the raw transcript</b></summary>

- **"Not enough GPU memory for LLM cleanup":** the cleanup model doesn't fit in free GPU
  memory, which is always the case in CPU mode. Close other GPU programs, pick a smaller model
  in the menu, or point `OLLAMA_URL` at a machine with more memory.
- **"LLM rewrite failed":** anything else, most often Ollama isn't running or the model isn't
  pulled. Check `ollama list` and `OLLAMA_URL`; the exact error is in the log
  (`journalctl --user -u tuxwhisper -n 20`).
</details>
