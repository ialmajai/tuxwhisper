# Other desktops

Bind these commands in your desktop's shortcut settings, using the full path to your clone:

| Key | Command |
|---|---|
| F5 | `~/tuxwhisper/asr toggle` |
| F3 | `~/tuxwhisper/asr rewrite` |
| F4 | `~/tuxwhisper/asr save` |
| Alt+F9 | `~/tuxwhisper/asr settings` |

Some desktops don't expand `~` in shortcut commands (GNOME's don't); write `/home/<you>/…` there.

| Desktop | Binding | Hold to talk |
|---|---|---|
| GNOME | `./asr bind`, or Settings → Keyboard → Custom Shortcuts | ✅ Yes |
| KDE Plasma | System Settings → Shortcuts → Add New → Command | ❔ Untested |
| Hyprland | `binde = , F5, exec, ~/tuxwhisper/asr toggle` | ❔ Untested (`binde` repeats while held, so holding should work) |
| Sway | `bindsym F5 exec ~/tuxwhisper/asr toggle` | ✅ Yes (tested on Sway 1.12) |
| i3 | `bindsym F5 exec --no-startup-id ~/tuxwhisper/asr toggle` | ✅ Yes (tested on i3 4.25) |

Where holding doesn't work, tap to start and tap again to stop.

The systemd service starts with `graphical-session.target`, which some window managers (i3,
or Sway without extra setup) never start. On those, `./install.sh` skips the service and says
so; by hand, skip step 3 of the [manual install](manual-install.md#3-start-the-daemon-at-login).
Start the daemon from your WM config instead, e.g. `exec ~/tuxwhisper/asr serve`.
