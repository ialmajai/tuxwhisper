# Security

> [!WARNING]
> Step 2 of the Quick start gives your user write access to `/dev/uinput`. TuxWhisper needs
> it to press the paste key, but it also lets **any program you run create a virtual keyboard
> or mouse and send input to any window**, including terminals and password prompts. Tools like
> `ydotool` require the same access.

The `uaccess` tag limits this to the user logged in at the active local session, and only
while that session is active. Other user accounts don't get access, but every process
running as you does, including ones started over SSH while you're logged in.

If you don't want this, skip step 2: TuxWhisper still transcribes, but it can't paste the text.
To undo it later, run `sudo rm /etc/udev/rules.d/60-uinput.rules` and reboot.
