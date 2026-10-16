#!/usr/bin/env bash
# TuxWhisper installer. Safe to run again: each step skips what is already done.
# Run it from a git clone (./install.sh), or as `tuxwhisper setup` after installing from PyPI.
# docs/manual-install.md does the same steps by hand.
#   --yes            answer yes to every question
#   --packages-only  install the system packages and stop (used by CI)
#   --engine=E       cuda (NVIDIA), vulkan (AMD/Intel GPUs) or cpu; detected when not given
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
# `tuxwhisper setup` runs this script from inside the installed package and passes its command;
# a git clone runs ./asr, which uses the clone's .venv.
PACKAGED="${TUXWHISPER_PACKAGED:-}"
CMD="${TUXWHISPER_CMD:-$DIR/asr}"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/256x256/apps"
RULE=/etc/udev/rules.d/60-uinput.rules
RULE_TEXT='KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"'
YES=0 PACKAGES_ONLY=0 engine=
for arg in "$@"; do
    case "$arg" in
        --yes) YES=1 ;;
        --packages-only) PACKAGES_ONLY=1 ;;
        --engine=cuda|--engine=vulkan|--engine=cpu) engine=${arg#--engine=} ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '  %s\n' "$*"; }
ask() { local a; [ "$YES" = 1 ] && return 0; read -r -p "  $1 [Y/n] " a; [[ -z $a || $a =~ ^[Yy] ]]; }

case "$DIR$CMD" in *[\"\|\&\\]*) echo "Move TuxWhisper to a path without \" | & or \\ characters." >&2; exit 1 ;; esac

# Speech engine: faster-whisper with CUDA on NVIDIA, whisper.cpp with Vulkan on AMD or Intel GPUs
# (PCI vendors 0x1002 and 0x8086), otherwise faster-whisper on the CPU.
if [ -z "$engine" ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then engine=cuda
    elif grep -qsxE '0x(1002|8086)' /sys/class/drm/card*/device/vendor; then engine=vulkan
    else engine=cpu; fi
fi

step "1/6  System packages"
# A compiler and kernel headers build the keyboard library; PortAudio reads the microphone.
# The rest are clipboard tools, the settings menu (yad adds hover help; openSUSE lacks it),
# notifications and the start/stop sounds.
# CI installs these lists on each distro; keep docs/manual-install.md in sync.
# shellcheck source=/dev/null
. /etc/os-release 2>/dev/null || true
case " ${ID:-} ${ID_LIKE:-} " in
    *" debian "*|*" ubuntu "*) pm=(apt-get install -y) pkgs=(libportaudio2 xclip wl-clipboard zenity yad libnotify-bin sound-theme-freedesktop)
        vulkan=(g++ cmake libvulkan-dev glslc spirv-headers glslang-tools mesa-vulkan-drivers) ;;
    *" fedora "*) pm=(dnf install -y) pkgs=(portaudio xclip wl-clipboard zenity yad libnotify sound-theme-freedesktop)
        vulkan=(gcc-c++ cmake vulkan-loader-devel glslc spirv-headers-devel glslang mesa-vulkan-drivers) ;;
    *" arch "*) pm=(pacman -S --needed --noconfirm) pkgs=(portaudio xclip wl-clipboard zenity yad libnotify sound-theme-freedesktop)
        vulkan=(gcc cmake vulkan-icd-loader vulkan-headers shaderc spirv-headers glslang vulkan-radeon vulkan-intel) ;;
    *" suse "*|*" opensuse "*) pm=(zypper -n install) pkgs=(portaudio xclip wl-clipboard zenity libnotify-tools sound-theme-freedesktop)
        vulkan=(gcc-c++ cmake vulkan-devel shaderc spirv-headers glslang-devel libvulkan_radeon libvulkan_intel) ;;
    *) pm=() pkgs=() vulkan=() ;;
esac
# The Vulkan engine is compiled on install: a C++ compiler, CMake, Vulkan headers and shader tools.
[ "$engine" = vulkan ] && pkgs+=("${vulkan[@]}")
have_all() {
    { command -v xclip || command -v wl-copy; } && command -v zenity && command -v notify-send &&
        { [[ " ${pkgs[*]} " != *" yad "* ]] || command -v yad; } &&
        [ -f /usr/share/sounds/freedesktop/stereo/device-added.oga ] &&
        { ldconfig -p 2>/dev/null || /sbin/ldconfig -p; } | grep libportaudio >/dev/null &&
        { [ "$engine" != vulkan ] || { command -v g++ && command -v cmake && command -v glslc &&
            command -v glslangValidator && [ -f /usr/include/vulkan/vulkan.h ]; }; }
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
case "$engine" in
    cuda) extra="[cuda]"; note "NVIDIA GPU: using the CUDA build." ;;
    vulkan) extra="[vulkan]"
        note "AMD or Intel GPU: using whisper.cpp with Vulkan, built here (a few minutes)."
        note "Untested on AMD and Intel so far; if it's slow, run this again with --engine=cpu." ;;
    cpu) extra=""; note "No supported GPU: using the CPU build (slower)." ;;
esac
build=(); [ "$engine" = vulkan ] && build=(env GGML_VULKAN=1)
if [ -n "$PACKAGED" ]; then
    # Installed with uv or pip: add the GPU libraries if this install lacks them.
    PY="$TUXWHISPER_PY"
    case "$engine" in
        cuda) check='import nvidia.cudnn, nvidia.cublas' ;;
        vulkan) check='import os, sysconfig; assert os.path.exists(sysconfig.get_paths()["purelib"] + "/libggml-vulkan.so")' ;;
        cpu) check='pass' ;;
    esac
    if "$PY" -c "$check" 2>/dev/null; then
        note "Already installed."
    else
        spec="tuxwhisper$extra==$TUXWHISPER_VERSION"
        nobin=(); [ "$engine" = vulkan ] && nobin=(--no-binary-package pywhispercpp)
        note "This runs: ${build[*]} uv tool install --reinstall ${nobin[*]} '$spec'"
        "${build[@]}" uv tool install -q --reinstall "${nobin[@]}" "$spec"
    fi
else
    [ -d .venv ] || uv venv -q --python 3.12 .venv
    uv pip uninstall -q --python .venv evdev 2>/dev/null || true  # replaced by evdev-binary in 0.2
    nobin=(); [ "$engine" = vulkan ] && nobin=(--no-binary pywhispercpp)
    "${build[@]}" uv pip install -q --python .venv "${nobin[@]}" -e ".$extra"
fi

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
    note "  exec \"$CMD\" serve"
    note "See docs/other-desktops.md."
else
    mkdir -p "$UNIT_DIR"
    if [ -f "$UNIT_DIR/tuxwhisper.service" ]; then
        note "Keeping your existing $UNIT_DIR/tuxwhisper.service."
    else
        sed "s|^ExecStart=.*|ExecStart=\"$CMD\" serve|" tuxwhisper.service > "$UNIT_DIR/tuxwhisper.service"
        note "Installed $UNIT_DIR/tuxwhisper.service."
    fi
    # Older unit files lack the restart limit, and existing ones are kept: add it as a drop-in.
    mkdir -p "$UNIT_DIR/tuxwhisper.service.d"
    printf '[Unit]\nStartLimitIntervalSec=300\nStartLimitBurst=3\n' > "$UNIT_DIR/tuxwhisper.service.d/restart-limit.conf"
    # One drop-in file names the engine; NVIDIA needs none (cuda is the default).
    rm -f "$UNIT_DIR/tuxwhisper.service.d/cpu.conf" "$UNIT_DIR/tuxwhisper.service.d/vulkan.conf"
    if [ "$engine" != cuda ]; then
        mkdir -p "$UNIT_DIR/tuxwhisper.service.d"
        printf '[Service]\nEnvironment=ASR_DEVICE=%s\n' "$engine" > "$UNIT_DIR/tuxwhisper.service.d/$engine.conf"
        note "Set ASR_DEVICE=$engine."
    fi
    systemctl --user daemon-reload
    systemctl --user enable -q tuxwhisper
    since="$(date '+%Y-%m-%d %H:%M:%S')"
    systemctl --user restart tuxwhisper
    note "Started; it also starts at every login."
fi

step "5/6  Keyboard shortcuts and app-grid launcher"
if [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]]; then
    "$CMD" bind | sed 's/^/  /' || note "Couldn't bind the keys; set them under Keys in the settings menu."
    note "Change keys under Keys in the settings menu. Defaults F5, F3 and F6 replace page refresh,"
    note "find-next and jump-to-address-bar in browsers; Ctrl+R, Ctrl+G and Ctrl+L still work."
else
    note "Bind these in your desktop's shortcut settings (docs/other-desktops.md):"
    note "  dictation: \"$CMD\" toggle     cleanup:  \"$CMD\" rewrite"
    note "  save:      \"$CMD\" save       settings: \"$CMD\" settings"
fi
mkdir -p "$APPS_DIR" "$ICON_DIR"
cp tuxwhisper.png "$ICON_DIR/tuxwhisper.png"
sed "s|@CMD@|$CMD|" tuxwhisper-settings.desktop > "$APPS_DIR/tuxwhisper-settings.desktop"
note "Added \"TuxWhisper Settings\" to your app grid."

step "6/6  Loading the speech model"
if [ "$service" = 0 ]; then
    note "Skipped: start TuxWhisper from your window manager (step 4), then press your dictation key."
    exit 0
fi
note "The first start downloads it (about 1.5 GB; the CPU model is smaller). Later starts take about 15 s."
for _ in $(seq 1 600); do
    log="$(journalctl --user -u tuxwhisper --since "$since" -o cat 2>/dev/null || true)"
    if grep -qx "ready" <<<"$log"; then
        step "Done. Click into any text box, hold the dictation key, wait for the start sound, speak, and release."
        note "Settings: the settings key, or \"TuxWhisper Settings\" in your apps.  LLM cleanup: docs/llm-cleanup.md"
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
