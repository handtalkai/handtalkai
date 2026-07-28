"""
camera_thread_test.py - Isolated test: does cv2.VideoCapture work when opened
and read from a background thread (like server.py does), on this machine?

This has nothing to do with Flask/SocketIO/AI models - it's purely checking
whether OpenCV + threading behaves correctly here, to confirm or rule out a
known Windows quirk.

Run with: python camera_thread_test.py
"""

import cv2
import threading
import time

CAMERA_INDEX = 0


def worker():
    print("[thread] Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    print("[thread] isOpened():", cap.isOpened())

    for i in range(30):
        ret, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"[thread] frame {i}: ret={ret}, shape={shape}")
        time.sleep(0.05)

    cap.release()
    print("[thread] Released camera.")


if __name__ == "__main__":
    print("[main] Starting background thread...")
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    print("[main] Done.")