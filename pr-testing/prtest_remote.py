"""Container process ownership helper; invoked through docker exec."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def proc_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        except PermissionError:
            environ = []
        return {
            "pid": int(pid),
            "state": fields[0],
            "pgid": int(fields[2]),
            "session": int(fields[3]),
            "start": fields[19],
            "token": next(
                (
                    v.split(b"=", 1)[1].decode()
                    for v in environ
                    if v.startswith(b"PRTEST_RUN_TOKEN=")
                ),
                None,
            ),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(info, identity):
    if not info or info["state"] == "Z":
        return False
    # Multiprocessing workers can clear their environment or rewrite argv/env.
    in_session = info["session"] == identity["pid"]
    has_token = info["token"] == identity["token"]
    return (in_session or has_token) and int(info["start"]) >= int(identity["start"])


def members(identity):
    if identity.get("session", identity["pid"]) != identity["pid"]:
        raise RuntimeError("invalid owned session identity")
    if (
        identity.get("boot_id")
        and identity["boot_id"]
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise RuntimeError("ownership record belongs to a different boot")
    leader = proc_info(identity["pid"])
    if leader and leader["start"] != identity["start"]:
        raise RuntimeError("PID reused; refusing to signal it")
    found = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            info = proc_info(path.name)
            if owned_process(info, identity):
                found.append(info)
    return found


def main():
    action, state_path = sys.argv[1:3]
    state = Path(state_path)
    if action == "launch":
        spec = json.loads(sys.argv[3])
        state.parent.mkdir(parents=True, exist_ok=True)
        if state.exists():
            raise RuntimeError("ownership file already exists")
        env = dict(os.environ, PRTEST_RUN_TOKEN=spec["token"])
        with open(spec["log"], "wb") as log, open(os.devnull, "rb") as stdin:
            child = subprocess.Popen(
                ["bash", "-c", spec["command"]],
                env=env,
                stdin=stdin,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        info = proc_info(child.pid)
        if not info:
            raise RuntimeError("server wrapper exited before ownership was recorded")
        identity = dict(
            info,
            token=spec["token"],
            command=spec["command"],
            log=spec["log"],
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
        temporary = state.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity))
        temporary.replace(state)
        print(json.dumps(identity))
        return
    identity = json.loads(state.read_text())
    if action == "check":
        print(json.dumps({"identity": identity, "members": members(identity)}))
    elif action == "stop":
        for sig, seconds in [(signal.SIGTERM, 15), (signal.SIGKILL, 5)]:
            # Inspect the session before each signal; never use a name sweep.
            processes = members(identity)
            for info in processes:
                current = proc_info(info["pid"])
                if (
                    current
                    and current["start"] == info["start"]
                    and owned_process(current, identity)
                ):
                    try:
                        os.kill(info["pid"], sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + seconds
            while members(identity) and time.monotonic() < deadline:
                time.sleep(0.1)
            if not members(identity):
                print(json.dumps({"stopped": True, "identity": identity}))
                return
        raise RuntimeError("owned processes survived SIGKILL")
    else:
        raise ValueError(action)


if __name__ == "__main__":
    main()
