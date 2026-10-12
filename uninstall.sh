#!/usr/bin/env bash
# Removes what install.sh added. Your settings, saved takes and the repo itself stay.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
RULE=/etc/udev/rules.d/60-uinput.rules

note() { printf '  %s\n' "$*"; }
ask() { local a; read -r -p "  $1 [y/N] " a; [[ $a =~ ^[Yy] ]]; }

systemctl --user disable -q --now tuxwhisper 2>/dev/null || true
rm -rf "$UNIT_DIR/tuxwhisper.service" "$UNIT_DIR/tuxwhisper.service.d"
systemctl --user daemon-reload
note "Removed the background service."

if command -v gsettings >/dev/null && [ -x "$DIR/.venv/bin/python" ] &&
   gsettings list-schemas | grep -qx org.gnome.settings-daemon.plugins.media-keys; then
    "$DIR/.venv/bin/python" - "$DIR" <<'EOF'
import ast, subprocess, sys

asr = sys.argv[1] + "/asr "
schema = "org.gnome.settings-daemon.plugins.media-keys"
def get(s, key):
    out = subprocess.run(["gsettings", "get", s, key], capture_output=True, text=True, check=True).stdout
    return ast.literal_eval(out.strip().removeprefix("@as "))
paths = get(schema, "custom-keybindings")
keep = [p for p in paths if not get(f"{schema}.custom-keybinding:{p}", "command").startswith(asr)]
for p in set(paths) - set(keep):
    subprocess.run(["gsettings", "reset-recursively", f"{schema}.custom-keybinding:{p}"], check=True)
subprocess.run(["gsettings", "set", schema, "custom-keybindings", repr(keep)], check=True)
print(f"  Removed {len(paths) - len(keep)} keyboard shortcut(s).")
EOF
fi

rm -f "$APPS_DIR/tuxwhisper-settings.desktop"
note "Removed the app-grid launcher."

if [ -f "$RULE" ] && ask "Remove the virtual keyboard rule ($RULE)? Other tools such as ydotool may use it."; then
    sudo rm "$RULE"
    note "Removed. It takes effect after a reboot."
fi

echo
note "Left in place, delete them yourself if you want:"
note "  $DIR  (the app, and recordings/ with any saved takes)"
note "  ${XDG_CONFIG_HOME:-$HOME/.config}/tuxwhisper  (your settings)"
note "  ~/.cache/huggingface  (downloaded models; other apps may share it)"
note "  ~/.local/share/pywhispercpp  (the Vulkan engine's model, if you used it)"
