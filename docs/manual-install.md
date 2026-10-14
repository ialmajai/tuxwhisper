# Manual install

`./install.sh` runs these same steps for you. Do them by hand if you want to run each command
yourself, use a desktop other than GNOME, or keep the repo somewhere other than `~/tuxwhisper`.

**Requirements:** Linux with systemd, [uv](https://docs.astral.sh/uv/) (it installs Python 3.12
for you), and a GPU with about 1.5 GB of free memory (NVIDIA, or AMD/Intel through
[Vulkan](configuration.md#amd-and-intel-gpus-vulkan)), or [CPU mode](configuration.md#cpu-mode). No uv yet? Its official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 0. Install system packages

PortAudio for the microphone, clipboard tools, `zenity` for the settings menu (`yad` adds hover
help and opens menus faster; openSUSE doesn't package it), and `notify-send` plus the sound theme for the recording
cues:

**Debian (Ubuntu uses the same packages):**

```bash
sudo apt install libportaudio2 \
  xclip wl-clipboard zenity yad libnotify-bin sound-theme-freedesktop
```

**Fedora:**

```bash
sudo dnf install portaudio \
  xclip wl-clipboard zenity yad libnotify sound-theme-freedesktop
```

**Arch:**

```bash
sudo pacman -S portaudio \
  xclip wl-clipboard zenity yad libnotify sound-theme-freedesktop
```

**openSUSE:**

```bash
sudo zypper install portaudio \
  xclip wl-clipboard zenity libnotify-tools sound-theme-freedesktop
```

These are the lists `install.sh` uses, and CI installs them on Debian, Fedora, Arch and
openSUSE. Without `notify-send` or the sounds, TuxWhisper still records; you just don't get
the cues.

## 1. Clone and install

```bash
git clone https://github.com/ialmajai/tuxwhisper.git ~/tuxwhisper
cd ~/tuxwhisper
uv venv --python 3.12 .venv
```

Then pick one. If `nvidia-smi` lists your GPU, use the first block. For an AMD or Intel GPU, use
[Vulkan](configuration.md#amd-and-intel-gpus-vulkan) (easiest with `./install.sh --engine=vulkan`).
Otherwise, the second block.

**NVIDIA GPU:**

```bash
uv pip install --python .venv -e '.[cuda]'
```

**No supported GPU (CPU mode):** this also tells the service, set up in step 3, to use the CPU.

```bash
uv pip install --python .venv -e .
mkdir -p ~/.config/systemd/user/tuxwhisper.service.d
printf '[Service]\nEnvironment=ASR_DEVICE=cpu\n' \
  > ~/.config/systemd/user/tuxwhisper.service.d/cpu.conf
```

To clone somewhere else, also edit `ExecStart` in `tuxwhisper.service` before step 3.

## 2. Allow the virtual keyboard

TuxWhisper types by pressing keys on a virtual keyboard, which needs this rule. The rule lets
**any program you run** send keystrokes, not only TuxWhisper; read the
[security note](security.md). Skip it and TuxWhisper copies each transcript for you to
paste with Ctrl+V instead. To undo it, delete the file and reboot.

```bash
echo 'KERNEL=="uinput", TAG+="uaccess", OPTIONS+="static_node=uinput"' |
  sudo tee /etc/udev/rules.d/60-uinput.rules
sudo udevadm control --reload && sudo udevadm trigger --name-match=uinput
getfacl /dev/uinput    # should list user:<your name>:rw-
```

## 3. Start the daemon at login

On i3, or Sway without a systemd session target, skip this step and start the daemon from your
window manager's config instead; see [Other desktops](other-desktops.md).

```bash
mkdir -p ~/.config/systemd/user
cp tuxwhisper.service ~/.config/systemd/user/
systemctl --user enable --now tuxwhisper
journalctl --user -u tuxwhisper -f    # wait for "ready", then press Ctrl+C
```

The first start downloads the Whisper model to `~/.cache/huggingface` (about 1.5 GB; the CPU
model is smaller), so it takes a while. After that, startup takes about 15 s.

## 4. Bind the keys

On GNOME, run this in the repo folder:

```bash
./asr bind
```

It binds the dictation, cleanup, save and settings keys as custom shortcuts (F5, F3, F4 and
Alt+F9). It keeps your other shortcuts, keeps any
key you already changed, and skips a key another shortcut uses. Change them later under
**Keys** in the settings menu, or in Settings → Keyboard → Keyboard Shortcuts → Custom Shortcuts. KDE, Hyprland, Sway, i3: see
[Other desktops](other-desktops.md).

The default dictation and cleanup keys, F5 and F3, replace page refresh and find-next in browsers
and many editors. Ctrl+R and Ctrl+G
still do those.

To open the settings menu from your app grid, run this in the repo folder to add
"TuxWhisper Settings":

```bash
sed "s|@DIR@|$PWD|" tuxwhisper-settings.desktop \
  > ~/.local/share/applications/tuxwhisper-settings.desktop
```

## 5. Try it

Click into any text box, hold the dictation key, wait for the start sound, speak, and release. Your words
appear at the cursor. Nothing typed? See [Troubleshooting](troubleshooting.md).

## 6. Optional: LLM cleanup

See [LLM cleanup](llm-cleanup.md#install).

## Uninstall by hand

`./uninstall.sh` does this for you.

```bash
systemctl --user disable --now tuxwhisper
rm -rf ~/.config/systemd/user/tuxwhisper.service ~/.config/systemd/user/tuxwhisper.service.d
rm -f ~/.local/share/applications/tuxwhisper-settings.desktop
systemctl --user daemon-reload
sudo rm /etc/udev/rules.d/60-uinput.rules    # takes effect after a reboot
```

Then remove the keyboard shortcuts and delete the repo folder. These stay until you delete
them: settings and lists in `~/.config/tuxwhisper`, recordings you saved with the save key in
`~/.local/share/tuxwhisper/recordings`, and the Whisper model in `~/.cache/huggingface/hub`.
