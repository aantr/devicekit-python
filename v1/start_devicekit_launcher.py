#!/usr/bin/env python3

import os
import shutil
import signal
import subprocess
import sys
import threading
import time

UDID = "00008140-001658323A29801C"
BUNDLE_ID = "com.xgame.devicekit-iosUITests.xctrunner"

HOST_PORT = 12005
PHONE_PORT = 12005


def find_ios() -> str:
    ios = shutil.which("ios")
    if not ios:
        print("ERROR: command 'ios' not found in PATH")
        print("Install go-ios first, for example:")
        print("  npm install -g go-ios")
        sys.exit(1)
    return ios


def stream_output(name: str, proc: subprocess.Popen):
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line}", end="")


def start_process(name: str, cmd: list[str]) -> subprocess.Popen:
    print()
    print(f"Starting {name}:")
    print("  " + " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    threading.Thread(
        target=stream_output,
        args=(name, proc),
        daemon=True,
    ).start()

    return proc


def stop_process(name: str, proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return

    print(f"Stopping {name}...")

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def check_process(name: str, proc: subprocess.Popen, delay: float = 0.8):
    time.sleep(delay)
    if proc.poll() is not None:
        raise RuntimeError(
            f"{name} exited early with code {proc.returncode}"
        )


def main():
    ios = find_ios()

    print("iPhone DeviceKit launcher")
    print("-------------------------")
    print(f"UDID      : {UDID}")
    print(f"Bundle ID : {BUNDLE_ID}")
    print()

    print("Requesting sudo permission for go-ios tunnel...")
    result = subprocess.run(["sudo", "-v"])

    if result.returncode != 0:
        print("ERROR: sudo authorization failed")
        sys.exit(result.returncode)

    tunnel = None
    devicekit = None
    forward = None

    try:
        tunnel = start_process(
            "tunnel",
            [
                "sudo",
                ios,
                "tunnel",
                "start",
                "--udid",
                UDID,
            ],
        )
        check_process("tunnel", tunnel, 1.0)

        # Give CoreDevice/RSD a moment to become available.
        time.sleep(1.5)

        devicekit = start_process(
            "devicekit",
            [
                ios,
                "ui",
                "run",
                "devicekit",
                "--bundleid",
                BUNDLE_ID,
                "--udid",
                UDID,
            ],
        )
        check_process("devicekit", devicekit, 1.0)

        time.sleep(1.0)

        forward = start_process(
            "h264",
            [
                ios,
                "forward",
                str(HOST_PORT),
                str(PHONE_PORT),
                "--udid",
                UDID,
            ],
        )
        check_process("h264 forward", forward, 0.8)

        print()
        print("All services started.")
        print()
        print("DeviceKit RPC:")
        print("  http://127.0.0.1:12004")
        print()
        print("ReplayKit H264:")
        print(f"  tcp://127.0.0.1:{HOST_PORT}")
        print()
        print("Now start Screen Broadcast on the iPhone.")
        print("Press Ctrl+C to stop everything.")
        print()

        while True:
            for name, proc in [
                ("tunnel", tunnel),
                ("devicekit", devicekit),
                ("h264 forward", forward),
            ]:
                rc = proc.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"{name} stopped unexpectedly with exit code {rc}"
                    )
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received.")

    except Exception as exc:
        print()
        print(f"ERROR: {exc}")

    finally:
        # Stop dependents before the tunnel.
        stop_process("h264 forward", forward)
        stop_process("devicekit", devicekit)
        stop_process("tunnel", tunnel)
        print("Stopped.")


if __name__ == "__main__":
    main()
