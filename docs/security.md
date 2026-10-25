# Security

> [!WARNING]
> The virtual keyboard rule, which `./install.sh` asks about (step 2 of the
> [manual install](manual-install.md#2-allow-the-virtual-keyboard)), gives your user write
> access to `/dev/uinput`. TuxWhisper needs it to press the paste key, but it also lets **any
> program you run create a virtual keyboard or mouse and send input to any window**, including
> terminals and password prompts. Tools like `ydotool` require the same access.

The `uaccess` tag limits this to the user logged in at the active local session, and only
while that session is active. Other user accounts don't get access, but every process
running as you does, including ones started over SSH while you're logged in.

If you don't want this, answer no when the installer asks, or skip step 2 by hand: TuxWhisper
then copies each transcript instead of typing it, and you paste it yourself with Ctrl+V.
To undo it later, run `./uninstall.sh`, or `sudo rm /etc/udev/rules.d/60-tuxwhisper-uinput.rules`, and
reboot.
