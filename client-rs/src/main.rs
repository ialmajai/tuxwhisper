//! TuxWhisper's key-press client. GNOME runs a command on every key press, and starting
//! Python for it takes 120-230 ms; this sends the key's command to the daemon's socket in
//! about a millisecond. Anything else (the daemon isn't running or still starting, no
//! desktop session) goes to the Python client, which knows what to do and say:
//! `$TUXWHISPER_CMD <command>`, with TUXWHISPER_SLOW=1 so the wrapper doesn't come back here.

use std::env;
use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{exit, Command};
use std::time::Duration;

const COMMANDS: [&str; 3] = ["toggle", "rewrite", "save"];

/// Whether the daemon got the command. Ok(false): it couldn't be reached or the command
/// couldn't be sent, so the Python client may safely try again. Once sent, the command is
/// never sent twice (a second toggle would undo the first), even if the reply is slow.
fn send(cmd: &str) -> io::Result<bool> {
    let Some(runtime) = env::var_os("XDG_RUNTIME_DIR") else {
        return Ok(false);
    };
    let Ok(mut sock) = UnixStream::connect(PathBuf::from(runtime).join("tuxwhisper.sock")) else {
        return Ok(false);
    };
    sock.set_write_timeout(Some(Duration::from_secs(5)))?;
    sock.set_read_timeout(Some(Duration::from_secs(5)))?;
    if sock
        .write_all(cmd.as_bytes())
        .and_then(|()| sock.shutdown(Shutdown::Write))
        .is_err()
    {
        return Ok(false);
    }
    let _ = sock.read(&mut [0u8; 16]); // the daemon answers by closing the connection
    Ok(true)
}

fn main() {
    let cmd = env::args().nth(1).unwrap_or_default();
    if COMMANDS.contains(&cmd.as_str()) && matches!(send(&cmd), Ok(true)) {
        return;
    }
    let Some(python_client) = env::var_os("TUXWHISPER_CMD") else {
        eprintln!("tuxwhisper-key: the daemon isn't reachable and TUXWHISPER_CMD isn't set");
        exit(1);
    };
    let err = Command::new(python_client)
        .args(env::args_os().skip(1))
        .env("TUXWHISPER_SLOW", "1")
        .exec();
    eprintln!("tuxwhisper-key: {err}");
    exit(1);
}
