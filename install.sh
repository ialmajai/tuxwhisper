#!/usr/bin/env bash
# TuxWhisper installer. Safe to run again: each step skips what is already done.
# docs/manual-install.md does the same steps by hand.
#   --yes            answer yes to every question
#   --packages-only  install the system packages and stop (used by CI)
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
RULE=/etc/udev/rules.d/60-uinput.rules
RULE_TEXT='KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"'
YES=0 PACKAGES_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --yes) YES=1 ;;
        --packages-only) PACKAGES_ONLY=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
ask() { local a; [ "$YES" = 1 ] && return 0; read -r -p "  $1 [Y/n] " a; [[ -z $a || $a =~ ^[Yy] ]]; }

case "$DIR" in *[\"\|\&\\]*) echo "Move the repo to a path without \" | & or \\ characters." >&2; exit 1 ;; esac

step "1/6  System packages"
# A compiler and kernel headers build the keyboard library; PortAudio reads the microphone.
# The rest are clipboard tools, the settings menu, notifications and the start/stop sounds.
# CI installs these lists on each distro; keep docs/manual-install.md in sync.
# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
case " ${ID:-} ${ID_LIKE:-} " in
    *" debian "*|*" ubuntu "*) pm=(apt-get install -y) pkgs=(gcc linux-libc-dev libportaudio2 xclip wl-clipboard zenity libnotify-bin sound-theme-freedesktop) ;;
    *" fedora "*) pm=(dnf install -y) pkgs=(gcc kernel-headers portaudio xclip wl-clipboard zenity libnotify sound-theme-freedesktop) ;;
    *" arch "*) pm=(pacman -S --needed --noconfirm) pkgs=(gcc linux-api-headers portaudio xclip wl-clipboard zenity libnotify sound-theme-freedesktop) ;;
    *" suse "*|*" opensuse "*) pm=(zypper -n install) pkgs=(gcc linux-glibc-devel portaudio xclip wl-clipboard zenity libnotify-tools sound-theme-freedesktop) ;;
    *) pm=() pkgs=() ;;
esac
have_all() {
    command -v gcc && command -v xclip && command -v wl-copy && command -v zenity && command -v notify-send &&
        [ -f /usr/include/linux/input.h ] && [ -f /usr/share/sounds/freedesktop/stereo/device-added.oga ] &&
        { ldconfig -p 2>/dev/null || /sbin/ldconfig -p; } | grep -q libportaudio
} >/dev/null 2>&1
if have_all; then
    note "Already installed."
elif [ ${#pm[@]} = 0 ]; then
    note "Unknown distro (${ID:-?}). Install the packages listed in docs/manual-install.md, then run this again."
    exit 1
else
    note "This runs: ${SUDO:+sudo }${pm[*]} ${pkgs[*]}"
    if ask "Install them?"; then
        [ "${pm[0]}" = apt-get ] && $SUDO apt-get update -q
        $SUDO "${pm[@]}" "${pkgs[@]}"
    else
        note "Skipped. Without them the Python packages may fail to build in the next step."
    fi
fi
[ "$PACKAGES_ONLY" = 1 ] && exit 0

# uv's installer puts uv in ~/.local/bin, which a shell opened before it isn't searching yet.
command -v uv >/dev/null || PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "TuxWhisper needs systemd. See docs/manual-install.md." >&2; exit 1; }

step "2/6  Python packages"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    gpu=1; reqs=requirements-cuda.txt; note "NVIDIA GPU found: installing the GPU build."
else
    gpu=0; reqs=requirements.txt; note "No NVIDIA GPU found: installing the CPU build (slower)."
fi
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv -r "$reqs"

step "3/6  Virtual keyboard (lets TuxWhisper type for you)"
if [ -f "$RULE" ] && grep -qF "$RULE_TEXT" "$RULE"; then
    note "Already set up."
else
    note "This writes one line to $RULE:"
    note "  $RULE_TEXT"
    note "It lets any program you run send keystrokes, not only TuxWhisper (docs/security.md)."
    if ask "Allow it? Needs your password once."; then
        echo "$RULE_TEXT" | $SUDO tee "$RULE" >/dev/null
        $SUDO udevadm control --reload
        $SUDO udevadm trigger --name-match=uinput
    else
        note "Skipped: TuxWhisper will copy each transcript for you to paste with Ctrl+V."
    fi
fi

step "4/6  Background service"
service=1
if ! systemctl --user is-active -q graphical-session.target; then
    service=0
    note "Your desktop doesn't start systemd's graphical session (common on i3 and Sway)."
    note "Start TuxWhisper from your window manager's config instead, e.g.:"
    note "  exec \"$DIR/asr\" serve"
    note "See docs/other-desktops.md."
else
    mkdir -p "$UNIT_DIR"
    if [ -f "$UNIT_DIR/tuxwhisper.service" ]; then
        note "Keeping your existing $UNIT_DIR/tuxwhisper.service."
    else
        sed "s|^ExecStart=.*|ExecStart=\"$DIR/asr\" serve|" tuxwhisper.service > "$UNIT_DIR/tuxwhisper.service"
        note "Installed $UNIT_DIR/tuxwhisper.service."
    fi
    if [ "$gpu" = 0 ]; then
        mkdir -p "$UNIT_DIR/tuxwhisper.service.d"
        printf '[Service]\nEnvironment=ASR_DEVICE=cpu\n' > "$UNIT_DIR/tuxwhisper.service.d/cpu.conf"
        note "Set CPU mode."
    fi
    systemctl --user daemon-reload
    systemctl --user enable -q tuxwhisper
    since="$(date '+%Y-%m-%d %H:%M:%S')"
    systemctl --user restart tuxwhisper
    note "Started; it also starts at every login."
fi

step "5/6  Keyboard shortcuts and app-grid launcher"
if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]]; then
    ./asr bind | sed 's/^/  /'
    note "F5 and F3 now replace page refresh and find-next in browsers; Ctrl+R and Ctrl+G still work."
else
    note "Bind these in your desktop's shortcut settings (docs/other-desktops.md):"
    note "  F5 -> \"$DIR/asr\" toggle      F3 -> \"$DIR/asr\" rewrite"
    note "  F4 -> \"$DIR/asr\" save        Alt+F9 -> \"$DIR/asr\" settings"
fi
mkdir -p "$APPS_DIR"
sed "s|@DIR@|$DIR|" tuxwhisper-settings.desktop > "$APPS_DIR/tuxwhisper-settings.desktop"
note "Added \"TuxWhisper Settings\" to your app grid."

step "6/6  Loading the speech model"
if [ "$service" = 0 ]; then
    note "Skipped: start TuxWhisper from your window manager (step 4), then press F5 to dictate."
    exit 0
fi
note "The first start downloads it (about 1.5 GB; the CPU model is smaller). Later starts take about 15 s."
for _ in $(seq 1 600); do
    log="$(journalctl --user -u tuxwhisper --since "$since" -o cat 2>/dev/null || true)"
    if grep -qx "ready" <<<"$log"; then
        step "Done. Click into any text box, hold F5, wait for the start sound, speak, and release."
        note "Settings: Alt+F9.  Optional LLM cleanup (F3): docs/llm-cleanup.md"
        exit 0
    fi
    if ! systemctl --user is-active -q tuxwhisper && [ -n "$log" ]; then
        echo "TuxWhisper stopped while starting. Last log lines:" >&2
        tail -n 5 <<<"$log" >&2
        echo "See docs/troubleshooting.md." >&2
        exit 1
    fi
    sleep 2
done
echo "Still loading after 20 minutes. Watch it with: journalctl --user -u tuxwhisper -f" >&2
exit 1
