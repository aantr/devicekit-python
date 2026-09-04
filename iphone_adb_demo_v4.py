import time

import cv2

from iphone_adb_fast_v4 import IPhoneADBFast


TARGET_FPS = 60.0

print("IMPORTANT:")
print("  1. ios ui run devicekit must be running")
print("  2. ios forward 12005 12005 must be running")
print("  3. Start Screen Broadcast on the iPhone")
print("  4. Close ffplay/nc before running this script")
print()

iphone = IPhoneADBFast(
    expected_video_fps=TARGET_FPS,
    auto_reset_backlog=True,
    backlog_fps_factor=1.60,
    socket_receive_buffer=256 * 1024,
    video_read_timeout=5.0,
)

print("Waiting for ReplayKit H264...")

frame, frame_id = iphone.wait_first_frame(
    timeout=30,
    copy=False,
)

print("Video connected.")
print("Device logical size:", iphone.screen_size)
print("ReplayKit frame:", frame.shape[1], "x", frame.shape[0])

display_frame = frame
display_frame_id = frame_id

last_tap = None
last_tap_at = 0.0


def mouse(event, x, y, flags, userdata):
    global last_tap
    global last_tap_at

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frame = display_frame
    if frame is None:
        return

    t0 = time.perf_counter()

    # IMPORTANT: do not execute synchronous XCTest RPC inside OpenCV's mouse
    # callback. cv2.waitKey()/imshow live on this same thread, so waiting for
    # the tap completion visually freezes the video window.
    future = iphone.tap_frame_async(
        x,
        y,
        frame,
    )

    last_tap = (x, y)
    last_tap_at = time.perf_counter()

    def tap_done(done_future):
        rpc_ms = (time.perf_counter() - t0) * 1000.0

        try:
            done_future.result()
            print(
                f"tap frame=({x},{y}) "
                f"RPC={rpc_ms:.1f} ms "
                f"(async; UI was not blocked)"
            )
        except Exception as exc:
            print(
                f"tap frame=({x},{y}) failed "
                f"after {rpc_ms:.1f} ms: {exc}"
            )

    future.add_done_callback(tap_done)


cv2.namedWindow("iPhone", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("iPhone", mouse)

try:
    last_shown_id = frame_id

    while True:
        try:
            frame, new_id = iphone.wait_next(
                last_shown_id,
                timeout=0.20,
                copy=False,
            )

            last_shown_id = new_id
            display_frame_id = new_id
            display_frame = frame

        except TimeoutError:
            frame = display_frame

        if frame is not None:
            # Only create a small display copy if we need to draw overlays.
            # This prevents modifying the shared CV frame.
            show = frame

            need_overlay = (
                last_tap is not None
                and time.perf_counter() - last_tap_at < 0.25
            )

            if need_overlay:
                show = frame.copy()

                cv2.circle(
                    show,
                    last_tap,
                    22,
                    (0, 0, 255),
                    3,
                )

                cv2.circle(
                    show,
                    last_tap,
                    4,
                    (0, 0, 255),
                    -1,
                )

            text = (
                f"{iphone.video_fps:.1f} fps  "
                f"frame {display_frame_id}  "
                f"age {(iphone.frame_age_ms or 0):.1f} ms  "
                f"state {iphone.video_state}  "
                f"resets {iphone.backlog_resets}"
            )

            # We need a copy only to render the status text.
            if show is frame:
                show = frame.copy()

            cv2.putText(
                show,
                text,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("iPhone", show)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord("h"):
            iphone.button("home")

        elif key == ord("r"):
            print("manual video reconnect")
            iphone.reconnect_video()

finally:
    iphone.close()
    cv2.destroyAllWindows()
