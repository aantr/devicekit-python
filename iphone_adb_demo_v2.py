import time

import cv2

from iphone_adb_fast_v2 import IPhoneADBFast


TARGET_FPS = 30.0

iphone = IPhoneADBFast(
    expected_video_fps=TARGET_FPS,
    auto_reset_backlog=True,
    backlog_fps_factor=1.60,
    socket_receive_buffer=256 * 1024,
)

frame, frame_id = iphone.wait_first_frame(
    timeout=10,
    copy=False,
)

print("Device logical size:", iphone.screen_size)
print("ReplayKit frame:", frame.shape[1], "x", frame.shape[0])

# Last frame actually shown in the window. Mouse clicks are converted
# against this frame, not an arbitrary newer frame.
display_frame = frame
display_frame_id = frame_id

last_tap = None
last_tap_at = 0.0


def mouse(event, x, y, flags, userdata):
    global last_tap, last_tap_at

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    frame = display_frame
    if frame is None:
        return

    # No wait_change() here: blocking an OpenCV mouse callback freezes
    # the preview and makes latency look much worse than it is.
    t0 = time.perf_counter()
    iphone.tap_frame(x, y, frame)
    rpc_ms = (time.perf_counter() - t0) * 1000.0

    last_tap = (x, y)
    last_tap_at = time.perf_counter()

    print(f"tap frame=({x},{y}) RPC={rpc_ms:.1f} ms")


cv2.namedWindow("iPhone", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("iPhone", mouse)

try:
    last_shown_id = frame_id

    while True:
        try:
            # Wait for a NEW frame instead of polling latest() thousands
            # of times per second. If several frames arrive while this
            # thread is busy, wait_next() returns the newest one.
            frame, new_id = iphone.wait_next(
                last_shown_id,
                timeout=0.20,
                copy=False,
            )
            last_shown_id = new_id
            display_frame_id = new_id
            display_frame = frame

        except TimeoutError:
            # Keep the window responsive if video pauses briefly.
            frame = display_frame

        if frame is not None:
            # Draw directly on this ndarray to avoid an 8-10 MB full-frame
            # copy on every iteration. The decoder publishes a new ndarray
            # for each next frame; it does not mutate this one afterward.
            if (
                last_tap is not None
                and time.perf_counter() - last_tap_at < 0.25
            ):
                cv2.circle(frame, last_tap, 22, (0, 0, 255), 3)
                cv2.circle(frame, last_tap, 4, (0, 0, 255), -1)

            age = iphone.frame_age_ms or 0.0
            text = (
                f"decode {iphone.video_fps:.1f} fps  "
                f"frame {display_frame_id}  "
                f"local-age {age:.1f} ms  "
                f"backlog-resets {iphone.backlog_resets}"
            )

            cv2.putText(
                frame,
                text,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("iPhone", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord("h"):
            iphone.button("home")
        elif key == ord("r"):
            print("manual video reconnect")
            iphone.reconnect_video()

finally:
    iphone.close()
    cv2.destroyAllWindows()
