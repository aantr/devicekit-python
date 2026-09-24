#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
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
LOG = logging.getLogger("devicekit_launcher")


def configure_logging(log_file: str):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        LOG.addHandler(handler)


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
    with proc.stdout:
        for line in proc.stdout:
            LOG.info("[%s] %s", name, line.rstrip())


def start_process(
    name: str,
    cmd: list[str],
    *,
    new_session: bool = True,
) -> subprocess.Popen:
    LOG.info("Starting %s: %s", name, " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdin=None,  # inherit current terminal stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=new_session,
    )

    threading.Thread(
        target=stream_output,
        args=(name, proc),
        daemon=True,
    ).start()

    return proc


def stop_process(
    name: str,
    proc: subprocess.Popen | None,
    *,
    process_group: bool = True,
):
    if proc is None or proc.poll() is not None:
        return

    LOG.info("Stopping %s...", name)

    try:
        if process_group:
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if process_group:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()


def check_process(
    name: str,
    proc: subprocess.Popen,
    delay: float = 0.8,
):
    time.sleep(delay)

    if proc.poll() is not None:
        raise RuntimeError(
            f"{name} exited early with code {proc.returncode}"
        )


@dataclass
class Service:
    name: str
    cmd: list[str]
    startup_delay: float
    new_session: bool = True
    proc: subprocess.Popen | None = None
    started_at: float | None = None


def stop_services(services: list[Service]):
    for service in reversed(services):
        stop_process(
            service.name, service.proc, process_group=service.new_session
        )
        service.proc = None
        service.started_at = None


def supervise(services: list[Service]):
    """Keep retrying; restart dependants when their upstream service exits."""
    retry_delay = 3
    healthy_since = None
    try:
        while True:
            for index, service in enumerate(services):
                try:
                    if service.proc is None:
                        service.proc = start_process(
                            service.name, service.cmd,
                            new_session=service.new_session,
                        )
                        service.started_at = time.monotonic()
                        check_process(
                            service.name, service.proc, service.startup_delay
                        )
                    rc = service.proc.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"{service.name} stopped unexpectedly with exit code {rc}"
                        )
                except (OSError, RuntimeError) as exc:
                    LOG.error("%s", exc)
                    if (
                        service.started_at is not None
                        and time.monotonic() - service.started_at > 3
                    ):
                        delay = 0
                        retry_delay = 3
                    else:
                        delay = retry_delay
                        retry_delay = min(retry_delay * 2, 30)
                    # Keep a healthy tunnel (and its sudo session) running when
                    # only DeviceKit or the video forwarder needs recovery.
                    stop_services(services[index:])
                    LOG.warning(
                        "Restarting %s and dependent services in %ss",
                        service.name, delay,
                    )
                    healthy_since = None
                    if delay:
                        time.sleep(delay)
                    break
            else:
                if healthy_since is None:
                    healthy_since = time.monotonic()
                    LOG.info("All services running. Press Ctrl+C to stop.")
                elif time.monotonic() - healthy_since >= 60:
                    retry_delay = 3
                time.sleep(1)
    finally:
        stop_services(services)


def request_stop(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(
        description="Start go-ios tunnel + DeviceKit + H264 forward"
    )
    parser.add_argument(
        "--userspace",
        action="store_true",
        help="Use go-ios userspace tunnel (no sudo required)",
    )
    parser.add_argument(
        "--log-file",
        default=str(Path(__file__).with_suffix(".log")),
        help="Rotating launcher and go-ios log file",
    )
    args = parser.parse_args()

    ios = find_ios()
    configure_logging(args.log_file)

    print("iPhone DeviceKit launcher")
    print("-------------------------")
    print(f"UDID      : {UDID}")
    print(f"Bundle ID : {BUNDLE_ID}")
    print(
        "Tunnel    : "
        + ("userspace (no sudo)" if args.userspace else "kernel (sudo)")
    )
    print()

    # tunnel is intentionally NOT put into a new session when sudo is used.
    # This keeps its controlling TTY, so sudo can authenticate normally.
    if args.userspace:
        tunnel_cmd = [ios, "tunnel", "start", "--userspace", "--udid", UDID]
    else:
        print("The tunnel command may ask for your macOS password.")
        tunnel_cmd = ["sudo", ios, "tunnel", "start", "--udid", UDID]

    services = [
        # Allow RSD/CoreDevice time to become available before starting DeviceKit.
        Service("tunnel", tunnel_cmd, 2.7, new_session=args.userspace),
        Service(
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
            2.2,
        ),
        Service(
            "h264 forward",
            [
                ios,
                "forward",
                str(HOST_PORT),
                str(PHONE_PORT),
                "--udid",
                UDID,
            ],
            0.8,
        ),
    ]

    LOG.info("DeviceKit RPC: http://127.0.0.1:12004")
    LOG.info("ReplayKit H264: tcp://127.0.0.1:%s", HOST_PORT)
    LOG.info("Start Screen Broadcast on the iPhone once services are running.")
    LOG.info("Automatic recovery enabled. Log: %s", args.log_file)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        supervise(services)
    except KeyboardInterrupt:
        LOG.info("Stop requested.")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        LOG.info("Stopped.")


if __name__ == "__main__":
    main()
