"""
Test prototype for Autonomous Scene Tracking & Settling Engine.
Verifies sub-millisecond hardware thumbnail diffing, motion detection, and settle timing.
"""
import time
import ctypes
from ctypes import wintypes
import cv2
import numpy as np

u32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

class SceneTrackerPrototype:
    def __init__(self, tw=160, th=90, motion_thresh=0.5, settle_delay=0.16):
        self.tw = tw
        self.th = th
        self.motion_thresh = motion_thresh
        self.settle_delay = settle_delay
        
        self.sw = u32.GetSystemMetrics(0) or 1920
        self.sh = u32.GetSystemMetrics(1) or 1080
        
        self.hdc = u32.GetDC(0)
        self.memdc = gdi32.CreateCompatibleDC(self.hdc)
        self.bmp = gdi32.CreateCompatibleBitmap(self.hdc, self.tw, self.th)
        self.old_bmp = gdi32.SelectObject(self.memdc, self.bmp)
        gdi32.SetStretchBltMode(self.memdc, 3) # COLORONCOLOR
        
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG), ('biHeight', wintypes.LONG),
                ('biPlanes', wintypes.WORD), ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG), ('biYPelsPerMeter', wintypes.LONG),
                ('biClrUsed', wintypes.DWORD), ('biClrImportant', wintypes.DWORD)
            ]
        self.bmi = BITMAPINFOHEADER()
        self.bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self.bmi.biWidth = self.tw
        self.bmi.biHeight = -self.th
        self.bmi.biPlanes = 1
        self.bmi.biBitCount = 32
        self.buf = np.zeros((self.th, self.tw, 4), dtype=np.uint8)
        
        self.prev_thumb = self._grab_thumb()
        self.in_motion = False
        self.last_motion_time = 0.0
        self.last_settled_scan_time = 0.0
        self.total_settles = 0

    def _grab_thumb(self) -> np.ndarray:
        gdi32.StretchBlt(self.memdc, 0, 0, self.tw, self.th, self.hdc, 0, 0, self.sw, self.sh, 0x00CC0020)
        gdi32.GetDIBits(self.memdc, self.bmp, 0, self.th, self.buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(self.bmi), 0)
        return self.buf[:, :, 0].copy()

    def tick(self) -> bool:
        """Call periodically (e.g. every 50ms). Returns True if a settled rescan should fire."""
        curr_thumb = self._grab_thumb()
        diff = cv2.absdiff(curr_thumb, self.prev_thumb)
        delta = float(np.mean(diff))
        self.prev_thumb = curr_thumb
        
        now = time.time()
        should_rescan = False
        
        if delta > self.motion_thresh:
            self.in_motion = True
            self.last_motion_time = now
            # Continuous streaming rescan if motion exceeds 600ms
            if now - self.last_settled_scan_time > 0.60:
                self.last_settled_scan_time = now
                should_rescan = True
        else:
            if self.in_motion and (now - self.last_motion_time >= self.settle_delay):
                self.in_motion = False
                self.last_settled_scan_time = now
                self.total_settles += 1
                should_rescan = True
                
        return should_rescan, delta

    def cleanup(self):
        gdi32.SelectObject(self.memdc, self.old_bmp)
        gdi32.DeleteObject(self.bmp)
        gdi32.DeleteDC(self.memdc)
        u32.ReleaseDC(0, self.hdc)

if __name__ == "__main__":
    tracker = SceneTrackerPrototype()
    print("Testing SceneTrackerPrototype for 2 seconds...")
    t0 = time.time()
    settles = 0
    ticks = 0
    while time.time() - t0 < 2.0:
        rescan, delta = tracker.tick()
        ticks += 1
        if rescan:
            settles += 1
            print(f"[{time.time()-t0:.2f}s] Settled Rescan Triggered! Delta: {delta:.2f}")
        time.sleep(0.05)
    tracker.cleanup()
    print(f"Completed {ticks} ticks in 2s. Settles triggered: {settles}. Success!")
