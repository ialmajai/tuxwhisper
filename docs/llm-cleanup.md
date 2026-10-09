# LLM cleanup (F3)

F3 sends the transcript to a local Ollama model (`llama3.2` by default). The model fixes
punctuation and capitalization and removes filler words ("um", "uh", "like"), repeated words
and false starts.

```text
Whisper:  um so can you like uh explain how the the attention mechanism works
Pasted:   So can you explain how the attention mechanism works?
```

- **Questions stay questions.** Dictate "what is a closure" and F3 pastes "What is a
  closure?" with no answer attached.
- **Same language out:** F3 answers in the language you spoke, in every mode. `llama3.2` is
  weak outside English (it garbles Arabic, for example). For other languages, `ollama pull
  qwen3:8b` (about 5 GB of GPU memory) and pick it as the LLM model in the menu.
- **Speed:** about 0.2–1 s once the model has loaded. The model unloads after 30 minutes
  idle to free GPU memory, so the next F3 takes a few seconds longer.
- **Always pastes something:** if Ollama is unreachable, or the model doesn't fit in free
  GPU memory, F3 pastes the raw transcript and a notification tells you why.
- **Not perfect:** it sometimes drops real words. Use F5 when the exact wording matters.

## Modes

Pick what F3 does in the settings menu (**F3 mode**). The recording notification shows the
active mode.

| Mode | What you get |
|---|---|
| Clean up (default) | Your words, with punctuation fixed and fillers removed |
| Fix grammar only | Grammar and punctuation fixed, your wording kept |
| Formal | A professional rewrite |
| Email | A short, polite email |
| Bullet points | A concise bullet list |

Add your own under **Custom F3 modes** in the menu, one `Name => instruction` per line:

```text
Tweet => Rewrite as a tweet under 280 characters.
Commit message => Rewrite as a one-line git commit message.
```

A custom mode with a built-in name replaces it. The rewriting modes change your wording, so
check the result before sending it.

Each mode can use its own model: pick the mode, then the **Model for …** row below it in the menu. For
example, keep the fast default for Clean up and use a bigger model for Email. Modes without
their own model use **LLM model (F3)**. A bigger model needs more GPU memory and takes longer.

Ollama can run on another machine; point `OLLAMA_URL` at it (see
[Configuration](configuration.md)).
