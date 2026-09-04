from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Optional, Tuple

import av
import cv2
import numpy as np
import requests
import websocket


class _VideoBacklogDetected(RuntimeError):
    pass


class IPhoneADBFast:
    """
    Low-latency iPhone automation client for DeviceKit.

    Control:
        ws://HOST:12004/ws      JSON-RPC over persistent WebSocket
        http://HOST:12004/rpc   HTTP fallback

    Video:
        HOST:12005              ReplayKit raw H.264 over TCP

    Low-latency behavior:
      * no decoded-frame queue: only the newest frame is retained
      * PyAV frame threading disabled
      * bounded TCP receive buffer
      * TCP_NODELAY
      * reconnect if the H264 connection is silent
      * reconnect if decoded FPS indicates stale backlog is being drained
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        rpc_port: int = 12004,
        video_port: int = 12005,
        rpc_timeout: float = 3.0,
        reconnect_delay: float = 0.10,
        auto_start_video: bool = True,
        expected_video_fps: float = 30.0,
        socket_receive_buffer: int = 256 * 1024,
        video_read_timeout: float = 5.0,
        auto_reset_backlog: bool = True,
        backlog_fps_factor: float = 1.60,
        backlog_grace_seconds: float = 1.50,
        fps_window_seconds: float = 0.50,
    ):
        self.host = host
        self.rpc_port = rpc_port
        self.video_port = video_port
        self.rpc_timeout = rpc_timeout
        self.reconnect_delay = reconnect_delay

        self.expected_video_fps = float(expected_video_fps)
        self.socket_receive_buffer = int(socket_receive_buffer)
        self.video_read_timeout = float(video_read_timeout)

        self.auto_reset_backlog = bool(auto_reset_backlog)
        self.backlog_fps_factor = float(backlog_fps_factor)
        self.backlog_grace_seconds = float(backlog_grace_seconds)
        self.fps_window_seconds = float(fps_window_seconds)

        self.ws_url = f"ws://{host}:{rpc_port}/ws"
        self.http_rpc_url = f"http://{host}:{rpc_port}/rpc"

        self._ws = None
        self._ws_lock = threading.Lock()
        self._rpc_id = 0
        self._http = requests.Session()

        self._screen_width: Optional[float] = None
        self._screen_height: Optional[float] = None
        self._screen_scale: Optional[float] = None

        self._frame_lock = threading.Lock()
        self._frame_cond = threading.Condition(self._frame_lock)

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_decode_time = 0.0

        self._video_fps = 0.0
        self._fps_window_started = time.perf_counter()
        self._fps_window_frames = 0

        self._stop = threading.Event()
        self._video_thread: Optional[threading.Thread] = None
        self._video_socket: Optional[socket.socket] = None

        self._video_error: Optional[str] = None
        self._video_state = "stopped"
        self._video_connected = False
        self._video_reconnects = 0
        self._backlog_resets = 0
        self._video_connection_started = 0.0

        self.connect_rpc()
        self.refresh_device_info()

        if auto_start_video:
            self.start_video()

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def connect_rpc(self) -> None:
        with self._ws_lock:
            self._close_ws_unlocked()
            self._ws = websocket.create_connection(
                self.ws_url,
                timeout=self.rpc_timeout,
                enable_multithread=True,
            )
            self._ws.settimeout(self.rpc_timeout)

    def _close_ws_unlocked(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _next_rpc_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _rpc_ws(self, method: str, params: Optional[dict] = None) -> Any:
        if params is None:
            params = {}

        with self._ws_lock:
            if self._ws is None:
                self._ws = websocket.create_connection(
                    self.ws_url,
                    timeout=self.rpc_timeout,
                    enable_multithread=True,
                )
                self._ws.settimeout(self.rpc_timeout)

            rpc_id = self._next_rpc_id()
            self._ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "method": method,
                        "params": params,
                    },
                    separators=(",", ":"),
                )
            )

            deadline = time.perf_counter() + self.rpc_timeout

            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"RPC timeout: {method}")

                self._ws.settimeout(remaining)
                raw = self._ws.recv()
                if raw is None:
                    raise ConnectionError("DeviceKit WebSocket closed")

                data = json.loads(raw)
                if data.get("id") != rpc_id:
                    continue

                if "error" in data:
                    raise RuntimeError(
                        f"DeviceKit RPC {method} failed: {data['error']}"
                    )

                return data.get("result")

    def _rpc_http(self, method: str, params: Optional[dict] = None) -> Any:
        rpc_id = self._next_rpc_id()
        response = self._http.post(
            self.http_rpc_url,
            json={
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": method,
                "params": params or {},
            },
            timeout=self.rpc_timeout,
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise RuntimeError(
                f"DeviceKit RPC {method} failed: {data['error']}"
            )
        return data.get("result")

    def rpc(self, method: str, params: Optional[dict] = None) -> Any:
        try:
            return self._rpc_ws(method, params)
        except Exception:
            with self._ws_lock:
                self._close_ws_unlocked()

            try:
                self.connect_rpc()
                return self._rpc_ws(method, params)
            except Exception:
                return self._rpc_http(method, params)

    # ------------------------------------------------------------------
    # Device info / controls
    # ------------------------------------------------------------------

    def refresh_device_info(self) -> dict:
        info = self.rpc("device.info")
        size = info.get("screenSize", {})
        self._screen_width = float(size["width"])
        self._screen_height = float(size["height"])
        self._screen_scale = float(info.get("scale", 1.0))
        return info

    @property
    def screen_size(self) -> Tuple[float, float]:
        if self._screen_width is None or self._screen_height is None:
            raise RuntimeError("Device screen size is not known")
        return self._screen_width, self._screen_height

    @property
    def screen_scale(self) -> float:
        return float(self._screen_scale or 1.0)

    def tap(self, x: float, y: float) -> Any:
        return self.rpc(
            "device.io.tap",
            {"x": float(x), "y": float(y)},
        )

    def swipe(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration: float = 0.10,
    ) -> Any:
        return self.rpc(
            "device.io.swipe",
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "duration": float(duration),
            },
        )

    def long_press(self, x: float, y: float, duration: float = 0.5) -> Any:
        return self.rpc(
            "device.io.longpress",
            {
                "x": float(x),
                "y": float(y),
                "duration": float(duration),
            },
        )

    def type_text(self, text: str) -> Any:
        return self.rpc("device.io.text", {"text": str(text)})

    def button(self, name: str) -> Any:
        return self.rpc("device.io.button", {"button": name})

    # ------------------------------------------------------------------
    # ReplayKit H264
    # ------------------------------------------------------------------

    def start_video(self) -> None:
        if self._video_thread is not None and self._video_thread.is_alive():
            return

        self._stop.clear()
        self._video_state = "starting"

        self._video_thread = threading.Thread(
            target=self._video_worker,
            name="iphone-replaykit-h264",
            daemon=True,
        )
        self._video_thread.start()

    def reconnect_video(self) -> None:
        sock = self._video_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    def _reset_fps_window(self) -> None:
        now = time.perf_counter()
        with self._frame_lock:
            self._fps_window_started = now
            self._fps_window_frames = 0
            self._video_fps = 0.0

    def _publish_frame(self, image: np.ndarray, now: float) -> float:
        with self._frame_cond:
            self._latest_frame = image
            self._frame_id += 1
            self._last_decode_time = now

            self._fps_window_frames += 1
            elapsed = now - self._fps_window_started

            if elapsed >= self.fps_window_seconds:
                self._video_fps = self._fps_window_frames / elapsed
                self._fps_window_frames = 0
                self._fps_window_started = now

            fps = self._video_fps
            self._video_state = "streaming"
            self._frame_cond.notify_all()

        return fps

    def _video_worker(self) -> None:
        while not self._stop.is_set():
            sock = None
            stream_file = None
            container = None
            connected_at = 0.0

            try:
                self._video_state = "connecting"
                self._video_error = None
                self._video_connected = False

                sock = socket.create_connection(
                    (self.host, self.video_port),
                    timeout=3.0,
                )

                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_RCVBUF,
                    self.socket_receive_buffer,
                )

                # Critical: never wait forever for a silent ReplayKit stream.
                sock.settimeout(self.video_read_timeout)

                self._video_socket = sock
                self._video_connected = True
                connected_at = time.perf_counter()
                self._video_connection_started = connected_at
                self._video_state = "connected_waiting_h264"

                self._reset_fps_window()

                stream_file = sock.makefile("rb", buffering=0)

                container = av.open(
                    stream_file,
                    mode="r",
                    format="h264",
                    options={
                        "fflags": "nobuffer",
                        "probesize": "32768",
                        "analyzeduration": "0",
                    },
                )

                video_stream = container.streams.video[0]
                video_stream.thread_type = "NONE"
                video_stream.thread_count = 1

                for packet in container.demux(video_stream):
                    if self._stop.is_set():
                        return

                    for frame in packet.decode():
                        if self._stop.is_set():
                            return

                        image = frame.to_ndarray(format="bgr24")
                        now = time.perf_counter()
                        fps = self._publish_frame(image, now)

                        if (
                            self.auto_reset_backlog
                            and self.expected_video_fps > 0
                            and now - connected_at >= self.backlog_grace_seconds
                            and fps > self.expected_video_fps * self.backlog_fps_factor
                        ):
                            self._backlog_resets += 1
                            raise _VideoBacklogDetected(
                                "H264 backlog detected: "
                                f"decoded {fps:.1f} fps, "
                                f"expected ~{self.expected_video_fps:.1f}"
                            )

                raise ConnectionError("H264 stream ended")

            except _VideoBacklogDetected as exc:
                self._video_error = str(exc)
                self._video_state = "backlog_reset"

                if not self._stop.is_set():
                    self._video_reconnects += 1
                    time.sleep(self.reconnect_delay)

            except (socket.timeout, TimeoutError) as exc:
                self._video_error = (
                    "ReplayKit TCP connected but no H264 data arrived "
                    f"for {self.video_read_timeout:.1f}s. "
                    "Start Screen Broadcast on iPhone and make sure "
                    "no ffplay/nc process is using port 12005."
                )
                self._video_state = "silent_reconnect"

                if not self._stop.is_set():
                    self._video_reconnects += 1
                    time.sleep(self.reconnect_delay)

            except Exception as exc:
                self._video_error = f"{type(exc).__name__}: {exc}"
                self._video_state = "error_reconnect"

                if not self._stop.is_set():
                    self._video_reconnects += 1
                    time.sleep(self.reconnect_delay)

            finally:
                self._video_connected = False

                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass

                if stream_file is not None:
                    try:
                        stream_file.close()
                    except Exception:
                        pass

                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass

                if self._video_socket is sock:
                    self._video_socket = None

    def wait_first_frame(
        self,
        timeout: float = 30.0,
        copy: bool = False,
    ) -> Tuple[np.ndarray, int]:
        deadline = time.perf_counter() + timeout

        with self._frame_cond:
            while self._latest_frame is None:
                remaining = deadline - time.perf_counter()

                if remaining <= 0:
                    raise TimeoutError(
                        "No ReplayKit frame received.\n"
                        f"video_state={self._video_state}\n"
                        f"tcp_connected={self._video_connected}\n"
                        f"reconnects={self._video_reconnects}\n"
                        f"last_error={self._video_error}\n\n"
                        "Check:\n"
                        "1) ios ui run devicekit is still running\n"
                        "2) ios forward 12005 12005 is still running\n"
                        "3) Screen Broadcast is ACTIVE on the iPhone\n"
                        "4) ffplay/nc is NOT connected to port 12005"
                    )

                self._frame_cond.wait(min(remaining, 0.5))

            frame = self._latest_frame.copy() if copy else self._latest_frame
            return frame, self._frame_id

    def latest(
        self,
        copy: bool = False,
    ) -> Tuple[Optional[np.ndarray], int]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None, self._frame_id

            frame = self._latest_frame.copy() if copy else self._latest_frame
            return frame, self._frame_id

    def wait_next(
        self,
        after_frame_id: int,
        timeout: float = 1.0,
        copy: bool = False,
    ) -> Tuple[np.ndarray, int]:
        deadline = time.perf_counter() + timeout

        with self._frame_cond:
            while self._frame_id <= after_frame_id:
                remaining = deadline - time.perf_counter()

                if remaining <= 0:
                    raise TimeoutError(
                        f"No frame after id={after_frame_id}. "
                        f"state={self._video_state}, "
                        f"error={self._video_error}"
                    )

                self._frame_cond.wait(remaining)

            frame = self._latest_frame.copy() if copy else self._latest_frame
            return frame, self._frame_id

    @property
    def frame_age_ms(self) -> Optional[float]:
        with self._frame_lock:
            t = self._last_decode_time

        if not t:
            return None

        return (time.perf_counter() - t) * 1000.0

    @property
    def video_fps(self) -> float:
        with self._frame_lock:
            return self._video_fps

    @property
    def video_error(self) -> Optional[str]:
        return self._video_error

    @property
    def video_state(self) -> str:
        return self._video_state

    @property
    def video_connected(self) -> bool:
        return self._video_connected

    @property
    def video_reconnects(self) -> int:
        return self._video_reconnects

    @property
    def backlog_resets(self) -> int:
        return self._backlog_resets

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def frame_to_device(
        self,
        x: float,
        y: float,
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        if frame is None:
            frame, _ = self.latest(copy=False)

        if frame is None:
            raise RuntimeError("No video frame available")

        frame_h, frame_w = frame.shape[:2]
        screen_w, screen_h = self.screen_size

        if (frame_w > frame_h) != (screen_w > screen_h):
            screen_w, screen_h = screen_h, screen_w

        return (
            float(x) * screen_w / float(frame_w),
            float(y) * screen_h / float(frame_h),
        )

    def device_to_frame(
        self,
        x: float,
        y: float,
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[int, int]:
        if frame is None:
            frame, _ = self.latest(copy=False)

        if frame is None:
            raise RuntimeError("No video frame available")

        frame_h, frame_w = frame.shape[:2]
        screen_w, screen_h = self.screen_size

        if (frame_w > frame_h) != (screen_w > screen_h):
            screen_w, screen_h = screen_h, screen_w

        return (
            int(round(float(x) * frame_w / screen_w)),
            int(round(float(y) * frame_h / screen_h)),
        )

    def tap_frame(
        self,
        x: float,
        y: float,
        frame: Optional[np.ndarray] = None,
    ) -> Any:
        dx, dy = self.frame_to_device(x, y, frame)
        return self.tap(dx, dy)

    # ------------------------------------------------------------------
    # Optional change detection
    # ------------------------------------------------------------------

    def wait_change(
        self,
        reference: np.ndarray,
        after_frame_id: int,
        timeout: float = 0.5,
        threshold: float = 4.0,
        downscale: float = 0.25,
    ) -> Tuple[np.ndarray, int, float]:
        ref = reference

        if downscale != 1.0:
            ref = cv2.resize(
                ref,
                None,
                fx=downscale,
                fy=downscale,
                interpolation=cv2.INTER_AREA,
            )

        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        deadline = time.perf_counter() + timeout
        current_id = after_frame_id

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("No visible frame change")

            frame, current_id = self.wait_next(
                current_id,
                timeout=remaining,
                copy=False,
            )

            test = frame

            if downscale != 1.0:
                test = cv2.resize(
                    test,
                    None,
                    fx=downscale,
                    fy=downscale,
                    interpolation=cv2.INTER_AREA,
                )

            if test.shape[:2] != ref_gray.shape[:2]:
                return frame, current_id, float("inf")

            test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
            diff = float(cv2.absdiff(ref_gray, test_gray).mean())

            if diff >= threshold:
                return frame, current_id, diff

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._stop.set()
        self._video_state = "stopping"

        sock = self._video_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

        if self._video_thread is not None:
            self._video_thread.join(timeout=2.0)

        with self._ws_lock:
            self._close_ws_unlocked()

        self._http.close()
        self._video_state = "stopped"

    def __enter__(self) -> "IPhoneADBFast":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
