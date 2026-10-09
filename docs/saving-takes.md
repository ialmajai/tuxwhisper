# Saving takes (F4)

F4 saves the most recent take to `recordings/`:

```text
recordings/
├── 20260101-120000.wav    # 16 kHz, mono, 16-bit
└── metadata.csv           # file_name,transcription
```

This is the Hugging Face `audiofolder` layout, ready for fine-tuning a speech model.

> [!IMPORTANT]
> The saved transcript must match exactly what you said, or a fine-tuned model learns the
> wrong words. If Whisper got it right, save it as is. If it got it wrong, save it and fix
> the line in `metadata.csv`: these corrected takes are the most valuable for fine-tuning,
> especially for accented speech.

- The notification shows your progress, e.g. "💾 Saved (142 takes, 14 min)". About an hour
  of audio is a good first target for fine-tuning.
- You can save only the **latest** take, and only once. Press F4 again and you get "Nothing
  to save" until you dictate again.
- The saved transcript is Whisper's **raw** output, even for F3 takes, because that's what
  matches the audio. F4 doesn't see corrections you make in the text box; edit
  `metadata.csv` instead.
