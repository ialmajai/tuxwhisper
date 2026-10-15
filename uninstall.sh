#!/usr/bin/env bash
# Removes what install.sh (or `tuxwhisper setup`) added. Your settings and saved recordings stay.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
RULE=/etc/udev/rules.d/60-uinput.rules
CMD="${TUXWHISPER_CMD:-$DIR/asr}"   # set by `tuxwhisper uninstall`
PY="${TUXWHISPER_PY:-$DIR/.venv/bin/python}"

note() { printf '  %s\n' "$*"; }
ask() { local a; read -r -p "  $1 [y/N] " a; [[ $a =~ ^[Yy] ]]; }

systemctl --user disable -q --now tuxwhisper 2>/dev/null || true
rm -rf "$UNIT_DIR/tuxwhisper.service" "$UNIT_DIR/tuxwhisper.service.d"
systemctl --user daemon-reload 2>/dev/null || true   # no user bus, e.g. over ssh
note "Removed the background service."

if command -v gsettings >/dev/null && [ -x "$PY" ] &&
   gsettings list-schemas | grep -x org.gnome.settings-daemon.plugins.media-keys >/dev/null; then
    "$PY" - "$CMD" <<'EOF'
import ast, os, shlex, subprocess, sys

COMMANDS = {"toggle", "rewrite", "save", "correct", "settings"}
def ours(command):
    # Exact, like the app's own matching: quoted paths count, other programs don't.
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return len(words) == 2 and os.path.basename(words[0]) in ("asr", "tuxwhisper") and words[1] in COMMANDS
schema = "org.gnome.settings-daemon.plugins.media-keys"
def get(s, key):
    out = subprocess.run(["gsettings", "get", s, key], capture_output=True, text=True, check=True).stdout
    return ast.literal_eval(out.strip().removeprefix("@as "))
paths = get(schema, "custom-keybindings")
keep = [p for p in paths if not ours(get(f"{schema}.custom-keybinding:{p}", "command"))]
for p in set(paths) - set(keep):
    subprocess.run(["gsettings", "reset-recursively", f"{schema}.custom-keybinding:{p}"], check=True)
subprocess.run(["gsettings", "set", schema, "custom-keybindings", repr(keep)], check=True)
print(f"  Removed {len(paths) - len(keep)} keyboard shortcut(s).")
EOF
fi

rm -f "$APPS_DIR/tuxwhisper-settings.desktop" "$ICON_DIR/tuxwhisper.png"
note "Removed the app-grid launcher."

if [ -f "$RULE" ] && ask "Remove the virtual keyboard rule ($RULE)? Other tools such as ydotool may use it."; then
    sudo rm "$RULE"
    note "Removed. It takes effect after a reboot."
fi

echo
note "Left in place, delete them yourself if you want:"
if [ -n "${TUXWHISPER_PACKAGED:-}" ]; then
    note "  the tuxwhisper command itself: uv tool uninstall tuxwhisper (or pip uninstall tuxwhisper)"
else
    note "  $DIR  (the app)"
fi
note "  ${XDG_CONFIG_HOME:-$HOME/.config}/tuxwhisper  (your settings, vocabulary, replacements, modes)"
note "  ${XDG_DATA_HOME:-$HOME/.local/share}/tuxwhisper  (recordings you saved with F4)"
note "  ~/.cache/huggingface  (downloaded models; other apps may share it)"
note "  ~/.local/share/pywhispercpp  (the Vulkan engine's model, if you used it)"
