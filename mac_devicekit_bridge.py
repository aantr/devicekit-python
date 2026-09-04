#!/usr/bin/env python3
"""
mac_devicekit_bridge.py

Thin TCP relay for DeviceKit:
    Mac 0.0.0.0:22004 -> 127.0.0.1:12004  (HTTP/WebSocket JSON-RPC)
    Mac 0.0.0.0:22005 -> 127.0.0.1:12005  (raw ReplayKit H264)

The bridge does NOT decode/re-encode video. It forwards bytes only.
"""

from __future__ import annotations

import argparse
import signal
import socket
import threading
import time
from dataclasses import dataclass


@dataclass
class RelayConfig:
    name: str
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int


class TCPRelay:
    def __init__(self, config: RelayConfig):
        self.config = config
        self._stop = threading.Event()
        self._listen_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._clients_lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._accept_loop,
            name=f"relay-{self.config.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

        if self._listen_socket is not None:
            try:
                self._listen_socket.close()
            except OSError:
                pass

        with self._clients_lock:
            clients = list(self._clients)

        for s in clients:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @staticmethod
    def _configure_socket(sock: socket.socket) -> None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass

        # Keep kernel buffering modest. We do not want a huge amount of
        # stale H264 sitting inside the Mac relay.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        except OSError:
            pass

        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
        except OSError:
            pass

    def _accept_loop(self) -> None:
        cfg = self.config

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_socket = srv

        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((cfg.listen_host, cfg.listen_port))
        srv.listen(16)
        srv.settimeout(1.0)

        print(
            f"[{cfg.name}] listening "
            f"{cfg.listen_host}:{cfg.listen_port} -> "
            f"{cfg.upstream_host}:{cfg.upstream_port}",
            flush=True,
        )

        while not self._stop.is_set():
            try:
                client, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            self._configure_socket(client)

            print(
                f"[{cfg.name}] client connected: "
                f"{addr[0]}:{addr[1]}",
                flush=True,
            )

            threading.Thread(
                target=self._handle_client,
                args=(client, addr),
                name=f"{cfg.name}-{addr[0]}:{addr[1]}",
                daemon=True,
            ).start()

    def _handle_client(self, client: socket.socket, addr) -> None:
        cfg = self.config
        upstream = None

        with self._clients_lock:
            self._clients.add(client)

        try:
            upstream = socket.create_connection(
                (cfg.upstream_host, cfg.upstream_port),
                timeout=3.0,
            )
            upstream.settimeout(None)
            self._configure_socket(upstream)

            with self._clients_lock:
                self._clients.add(upstream)

            # Full duplex proxy. One thread per direction keeps the bridge
            # extremely simple and avoids application-level queues.
            t1 = threading.Thread(
                target=self._pipe,
                args=(client, upstream),
                name=f"{cfg.name}-client-to-device",
                daemon=True,
            )
            t2 = threading.Thread(
                target=self._pipe,
                args=(upstream, client),
                name=f"{cfg.name}-device-to-client",
                daemon=True,
            )

            t1.start()
            t2.start()

            # Wait until one side closes, then tear down both sockets.
            while (
                not self._stop.is_set()
                and t1.is_alive()
                and t2.is_alive()
            ):
                time.sleep(0.05)

        except Exception as exc:
            print(
                f"[{cfg.name}] connection error: {exc}",
                flush=True,
            )

        finally:
            for s in (client, upstream):
                if s is None:
                    continue

                with self._clients_lock:
                    self._clients.discard(s)

                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

                try:
                    s.close()
                except OSError:
                    pass

            print(
                f"[{cfg.name}] client disconnected: "
                f"{addr[0]}:{addr[1]}",
                flush=True,
            )

    def _pipe(self, src: socket.socket, dst: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                data = src.recv(64 * 1024)

                if not data:
                    return

                dst.sendall(data)

        except (ConnectionResetError, BrokenPipeError, OSError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose local DeviceKit ports to Windows/LAN."
    )

    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="Interface to listen on. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=22004,
        help="External control/RPC port. Default: 22004",
    )
    parser.add_argument(
        "--video-port",
        type=int,
        default=22005,
        help="External raw H264 port. Default: 22005",
    )

    args = parser.parse_args()

    control = TCPRelay(
        RelayConfig(
            name="control",
            listen_host=args.bind,
            listen_port=args.control_port,
            upstream_host="127.0.0.1",
            upstream_port=12004,
        )
    )

    video = TCPRelay(
        RelayConfig(
            name="video",
            listen_host=args.bind,
            listen_port=args.video_port,
            upstream_host="127.0.0.1",
            upstream_port=12005,
        )
    )

    stop_event = threading.Event()

    def request_stop(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print("DeviceKit Mac bridge")
    print("--------------------")
    print("Make sure these are already running on the Mac:")
    print("  ios tunnel start ...")
    print("  ios ui run devicekit ...")
    print("  ios forward 12005 12005 ...")
    print()
    print("Press Ctrl+C to stop.")
    print()

    control.start()
    video.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.25)
    finally:
        print("\nStopping bridge...")
        video.stop()
        control.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
