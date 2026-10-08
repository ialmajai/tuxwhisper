# Other desktops

Bind the same commands in your desktop's shortcut settings:

| Desktop | Binding | Hold to talk |
|---|---|---|
| GNOME | Settings → Keyboard → Custom Shortcuts | ✅ Yes |
| KDE Plasma | System Settings → Shortcuts → Add New → Command | ❔ Untested |
| Hyprland | `binde = , F5, exec, ~/tuxwhisper/asr toggle` | ✅ Should work (`binde` repeats while held) |
| Sway | `bindsym F5 exec ~/tuxwhisper/asr toggle` | ❌ Tap only |
| i3 | `bindsym F5 exec --no-startup-id ~/tuxwhisper/asr toggle` | ❌ Tap only |

Where holding doesn't work, tap to start and tap again to stop.

The systemd service starts with `graphical-session.target`, which some window managers (i3,
or Sway without extra setup) never start. On those, skip step 3 of the Quick start
and start the daemon from your WM config instead, e.g. `exec ~/tuxwhisper/asr serve`.
