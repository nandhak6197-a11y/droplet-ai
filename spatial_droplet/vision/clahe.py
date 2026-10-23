"""
Adaptive Contrast Normalizer (CLAHE).
Eliminates contrast fragility across Windows Dark Mode, pastel themes, and flat borders.
Expands local dynamic range before Canny edge and contour detection passes.
"""
import cv2
import numpy as np


class AdaptiveContrastNormalizer:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    along with dynamic range expansion to boost edge contrast on subtle,
    low-contrast, or dark mode desktop controls.
    """
    _clahe = None

    @classmethod
    def get_clahe(cls):
        if cls._clahe is None:
            cls._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return cls._clahe

    @classmethod
    def enhance_grayscale(cls, gray: np.ndarray) -> np.ndarray:
        """Applies adaptive dynamic range expansion + CLAHE to a grayscale image."""
        if gray.size == 0:
            return gray

        g_min = int(np.min(gray))
        g_max = int(np.max(gray))

        # When dynamic range is compressed (< 180), adaptively stretch to full range
        if g_max > g_min and (g_max - g_min) < 180:
            stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        else:
            stretched = gray

        clahe = cls.get_clahe()
        return clahe.apply(stretched)

    @classmethod
    def normalize_grayscale(cls, img: np.ndarray) -> np.ndarray:
        """Converts BGR or grayscale image to CLAHE contrast-normalized grayscale."""
        if img.size == 0:
            return img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cls.enhance_grayscale(gray)

    @classmethod
    def enhance_bgr_luminance(cls, bgr: np.ndarray) -> np.ndarray:
        """Equalizes luminance channel in LAB color space, preserving natural color."""
        if bgr.size == 0:
            return bgr
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced_l = cls.enhance_grayscale(l)
        merged = cv2.merge((enhanced_l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
