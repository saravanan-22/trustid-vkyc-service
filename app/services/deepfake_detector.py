import cv2
import numpy as np
from typing import Tuple
from app.core.config import settings
from app.schemas.verification import DeepfakeMetrics


class DeepfakeDetector:
    """
    Heuristic synthetic-media signals from 2D FFT spectral statistics and chroma
    noise.

    IMPORTANT: this is NOT a trained deepfake classifier and has not been
    evaluated against modern face-swap or diffusion output. Its signals are used
    only to route a verification to manual review, never to approve one. Replace
    it with a trained, benchmarked model before relying on it.
    """

    def analyze_frequency_spectrum(self, img_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Calculates azimuthal radial power spectrum using 2D Fast Fourier Transform.
        Natural optical sensors exhibit characteristic 1/f^alpha spectral roll-off.
        Generative models (GANs/Diffusion/DeepFaceLab) generate high-frequency
        periodic grid anomalies and irregular high-frequency energy ratios.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Resize to fixed dimension for normalized spectrum calculation
        dim = 256
        gray_resized = cv2.resize(gray, (dim, dim), interpolation=cv2.INTER_AREA)

        # 2D Fast Fourier Transform
        f = np.fft.fft2(gray_resized.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) + 1e-9

        # Azimuthal average / Radial profile
        y, x = np.ogrid[:dim, :dim]
        center = dim // 2
        r = np.hypot(x - center, y - center).astype(np.int32)

        # High-frequency vs Mid-frequency energy ratio
        high_freq_mask = (r > (dim * 0.35)) & (r < center)
        mid_freq_mask = (r > (dim * 0.10)) & (r <= (dim * 0.35))

        high_energy = np.mean(magnitude_spectrum[high_freq_mask])
        mid_energy = np.mean(magnitude_spectrum[mid_freq_mask])

        # High-frequency anomaly ratio
        # Pure digital/synthetic frames have abnormal ratios due to upsampling filters
        ratio = high_energy / (mid_energy + 1e-6)

        # Variance of the high-frequency spectrum (checkerboard artifacts induce high local variance)
        spectral_variance = float(np.var(magnitude_spectrum[high_freq_mask]) / (np.mean(magnitude_spectrum[high_freq_mask]) ** 2 + 1e-6))
        
        # Normalize into a risk score between 0.0 and 1.0.
        # Genuine photographs measured during development had ratios of roughly
        # 0.29-0.42 and spectral variance below 0.8. The bounds are configurable
        # and loosely calibrated; treat the result as a weak signal.
        anomaly_score = 0.0
        if ratio > settings.FFT_RATIO_MAX or ratio < settings.FFT_RATIO_MIN:
            anomaly_score += 0.45
        if spectral_variance > settings.FFT_SPECTRAL_VARIANCE_MAX:
            anomaly_score += 0.40

        risk_score = float(np.clip(anomaly_score, 0.0, 1.0))
        return risk_score, spectral_variance

    def detect_virtual_camera(self, img_bgr: np.ndarray) -> bool:
        """
        Examines frame characteristics common in virtual camera injection (e.g. OBS, ManyCam).
        Checks for artificial chroma uniformity, perfect zero-noise blocks, or flat histogram peaks.
        """
        # Convert to YUV to analyze noise in chrominance channels
        yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        u_channel = yuv[:, :, 1]
        v_channel = yuv[:, :, 2]

        u_std = np.std(u_channel)
        v_std = np.std(v_channel)

        # In natural camera sensor capture, sensor thermal noise creates standard deviation > 1.5 in chroma
        # Completely synthetic or frozen virtual inputs often exhibit unnaturally flat chroma variance
        if u_std < 1.0 and v_std < 1.0:
            return True

        return False

    def evaluate_frame(self, img_bgr: np.ndarray) -> DeepfakeMetrics:
        """
        Runs the full deepfake and virtual camera detection suite on a frame.
        """
        risk_score, spectral_variance = self.analyze_frequency_spectrum(img_bgr)
        virtual_cam = self.detect_virtual_camera(img_bgr)

        if virtual_cam:
            risk_score = min(1.0, risk_score + 0.5)

        is_suspected = risk_score > settings.DEEPFAKE_MAX_RISK_SCORE

        return DeepfakeMetrics(
            synthetic_risk_score=round(risk_score, 4),
            is_deepfake_suspected=is_suspected,
            virtual_camera_detected=virtual_cam,
            spectral_artifact_score=round(spectral_variance, 4),
        )


deepfake_detector = DeepfakeDetector()
