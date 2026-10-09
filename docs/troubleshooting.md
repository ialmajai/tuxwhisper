# Troubleshooting

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
<summary><b>It transcribes but doesn't paste</b></summary>

Run `getfacl /dev/uinput`; it should list `user:<you>:rw-`. If not, check the udev rule from
step 2 of the Quick start, including the file name (it must start with a number below 73).
</details>

<details>
<summary><b>It doesn't paste in one app</b></summary>

That app may not treat Shift+Insert as paste. Set **Paste key** to `ctrl+v` in the settings
menu.
</details>

<details>
<summary><b>It pastes my old clipboard instead of the transcript</b></summary>

The app read the clipboard after TuxWhisper had restored it. Increase the `0.3` s delay in
`paste()` in `asr.py`.
</details>

<details>
<summary><b>The daemon exits with "cuda unavailable"</b></summary>

TuxWhisper couldn't use the GPU or the CUDA libraries. Check that you installed from
`requirements-cuda.txt` and that `nvidia-smi` works, or switch to [CPU mode](configuration.md#cpu-mode).
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
