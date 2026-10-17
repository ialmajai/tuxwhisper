# Saving recordings

The save key (F4) shows the most recent recording's transcript and asks before saving. Fix
any wrong words, then press **Save**, or **Cancel** to keep nothing. Saved recordings go to
`~/.local/share/tuxwhisper/recordings/`:

```text
recordings/
├── 20260101-120000.wav    # 16 kHz, mono, 16-bit
└── metadata.csv           # file_name,transcription
```

This is the Hugging Face `audiofolder` layout, ready for fine-tuning a speech model.

> [!IMPORTANT]
> The saved transcript must match exactly what you said, or a fine-tuned model learns the
> wrong words. If Whisper got it wrong, fix it in the save dialog before pressing **Save**.
> These corrected recordings are the most valuable for fine-tuning, especially for accented
> speech. To fix one later, use **Correct last transcript** and tick "Fix the saved
> recording's transcript", or edit its line in `metadata.csv`.

- The notification shows your progress, e.g. "Saved (142 recordings, 14 min)". About an hour
  of audio is a good first target for fine-tuning.
- You can save only the **latest** recording, and only once. Press the save key again and you get "Nothing
  to save" until you dictate again. After **Cancel**, the save key offers the same recording again.
- Dictation keeps working while the save dialog is open.
- The save dialog shows Whisper's **plain** output, even for recordings with LLM cleanup, because that's what
  matches the audio. It doesn't see corrections you make in the app you dictated into.
