# Other desktops

Bind these commands to keys of your choice in your desktop's shortcut settings. Installed from
PyPI, the command is `~/.local/bin/tuxwhisper`; in a git clone, it's `~/tuxwhisper/asr`:

| Action | Command |
|---|---|
| Dictate | `~/.local/bin/tuxwhisper toggle` |
| Dictate with LLM cleanup | `~/.local/bin/tuxwhisper rewrite` |
| Save the last recording | `~/.local/bin/tuxwhisper save` |
| Open the settings menu | `~/.local/bin/tuxwhisper settings` |

On GNOME, setup binds them to F5, F3, F4 and Alt+F9; the examples below use F5 for dictation.

Some desktops don't expand `~` in shortcut commands (GNOME's don't); write `/home/<you>/…` there.

| Desktop | Binding | Hold to talk |
|---|---|---|
| GNOME | `tuxwhisper setup` binds them; change them under **Keys** in the settings menu | ✅ Yes |
| KDE Plasma | System Settings → Shortcuts → Add New → Command | ❔ Untested |
| Hyprland | `binde = , F5, exec, ~/tuxwhisper/asr toggle` | ❔ Untested (`binde` repeats while held, so holding should work) |
| Sway | `bindsym F5 exec ~/tuxwhisper/asr toggle` | ✅ Yes (tested on Sway 1.12) |
| i3 | `bindsym F5 exec --no-startup-id ~/tuxwhisper/asr toggle` | ✅ Yes (tested on i3 4.25) |

Where holding doesn't work, tap to start and tap again to stop.

The systemd service starts with `graphical-session.target`, which some window managers (i3,
or Sway without extra setup) never start. On those, `./install.sh` skips the service and says
so; by hand, skip step 3 of the [manual install](manual-install.md#3-start-the-daemon-at-login).
Start the daemon from your WM config instead, e.g. `exec ~/tuxwhisper/asr serve`.
