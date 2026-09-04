import time

import cv2

from iphone_adb_fast import IPhoneADBFast


iphone = IPhoneADBFast()

frame, frame_id = iphone.wait_first_frame(timeout=10)

print("Device logical size:", iphone.screen_size)
print("ReplayKit frame:", frame.shape[1], "x", frame.shape[0])

last_tap = None
last_tap_at = 0.0


def mouse(event, x, y, flags, userdata):
    global last_tap, last_tap_at

    if event == cv2.EVENT_LBUTTONDOWN:
        before, before_id = iphone.latest(copy=True)

        t0 = time.perf_counter()
        iphone.tap_frame(x, y, before)

        last_tap = (x, y)
        last_tap_at = time.perf_counter()

        # Optional: estimate tap -> first visible change.
        try:
            _, changed_id, diff = iphone.wait_change(
                before,
                after_frame_id=before_id,
                timeout=0.5,
                threshold=4.0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            print(
                f"tap ({x},{y}) -> frame {changed_id}, "
                f"visible change {latency_ms:.1f} ms, diff={diff:.2f}"
            )
        except TimeoutError:
            print(f"tap ({x},{y}), no visible change in 500 ms")


cv2.namedWindow("iPhone", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("iPhone", mouse)

try:
    while True:
        frame, frame_id = iphone.latest(copy=True)
        if frame is None:
            continue

        # Visual tap marker on the computer preview.
        if last_tap is not None and time.perf_counter() - last_tap_at < 0.25:
            cv2.circle(frame, last_tap, 22, (0, 0, 255), 3)
            cv2.circle(frame, last_tap, 4, (0, 0, 255), -1)

        text = (
            f"FPS {iphone.video_fps:.1f}  "
            f"frame {frame_id}  "
            f"age {iphone.frame_age_ms or 0:.1f} ms"
        )
        cv2.putText(
            frame,
            text,
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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

finally:
    iphone.close()
    cv2.destroyAllWindows()
