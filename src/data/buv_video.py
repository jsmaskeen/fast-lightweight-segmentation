import shutil
from pathlib import Path
from typing import Generator, Tuple, Optional
import cv2
import numpy as np
from huggingface_hub import hf_hub_download


def prepare_buv_videos(
    sample_filename: str = "videos_full/class_0/0969228d24724272bc3ee4b499b39cb8.avi"
) -> Path:
    """
    Downloads and caches a real clinical breast ultrasound video from
    the MICCAI 2022 BUV / MIBUS benchmark repository on Hugging Face (xenosscu/BUVFM).
    """
    cine_dir = Path("data/cine_videos")
    cine_dir.mkdir(parents=True, exist_ok=True)
    target_path = cine_dir / "buv_clinical_sample.avi"

    if target_path.exists() and target_path.stat().st_size > 1_000_000:
        return target_path

    print(f"Downloading clinical ultrasound sequence from xenosscu/BUVFM ({sample_filename})...")
    cached_path = hf_hub_download(
        repo_id="xenosscu/BUVFM",
        filename=sample_filename,
        repo_type="dataset",
    )
    shutil.copyfile(cached_path, target_path)
    print(f"Clinical video sequence cached at: {target_path} ({target_path.stat().st_size / 1e6:.2f} MB)")
    return target_path


class ClinicalUltrasoundStream:
    """
    Streams frames from a real clinical ultrasound cine-loop (.avi).
    Standardizes each frame to 256x256 RGB normalized float32 tensor [1, 3, 256, 256]
    using ImageNet MEAN/STD matching model training.
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, video_path: Optional[Path] = None, loop: bool = True):
        if video_path is None or not Path(video_path).exists():
            video_path = prepare_buv_videos()
        self.video_path = Path(video_path)
        self.loop = loop
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Loaded clinical ultrasound video: {self.width}x{self.height}, {self.frame_count} frames at {self.fps:.1f} FPS")

    def __iter__(self) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break

            # Raw frame display (BGR)
            raw_display = cv2.resize(frame, (256, 256))

            # Preprocessing with standard ImageNet normalization
            rgb = cv2.cvtColor(raw_display, cv2.COLOR_BGR2RGB)
            norm = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
            tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 256, 256]

            yield frame_idx, raw_display, tensor
            frame_idx += 1

    def release(self):
        if self.cap:
            self.cap.release()


class POCUSContinuousSweepStream:
    """
    Synthesizes a realistic continuous POCUS probe transducer sweep across
    confirmed clinical lesion scans (benign/malignant).
    Simulates probe translation, breathing micro-heave, and speckle noise at 30 FPS.
    Yields (frame_idx, raw_display_bgr, tensor_norm, gt_mask_bin).
    """
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, case_id: str = "benign (10)", num_frames: int = 120, fps: float = 30.0):
        img_p = Path(f"data/processed/images/{case_id}.png")
        mask_p = Path(f"data/processed/masks/{case_id}.png")
        if not img_p.exists():
            img_p = list(Path("data/processed/images").glob("benign*.png"))[0]
            mask_p = Path("data/processed/masks") / img_p.name

        self.img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        self.mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        self.num_frames = num_frames
        self.fps = fps
        self.h, self.w = 256, 256

    def __iter__(self) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray], None, None]:
        for f in range(self.num_frames):
            # Probe panning translation (smooth sinusoidal trajectory)
            dx = 18.0 * np.sin(2.0 * np.pi * f / 60.0)
            dy = 10.0 * np.cos(2.0 * np.pi * f / 60.0)
            # Breathing expansion / contraction zoom (1.0 to 1.04)
            scale = 1.0 + 0.03 * np.sin(2.0 * np.pi * f / 45.0)

            # Affine transformation matrix
            center = (self.w / 2, self.h / 2)
            M = cv2.getRotationMatrix2D(center, 0.0, scale)
            M[0, 2] += dx
            M[1, 2] += dy

            # Warp image and mask
            warped_img = cv2.warpAffine(self.img, M, (self.w, self.h), borderMode=cv2.BORDER_REFLECT_101)
            warped_mask = cv2.warpAffine(self.mask, M, (self.w, self.h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_mask = (warped_mask > 127).astype(np.uint8) * 255

            # Subtle speckle fluctuation
            noise = np.random.normal(0, 3.0, (self.h, self.w)).astype(np.float32)
            noisy_img = np.clip(warped_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            raw_display = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
            norm = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
            tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

            yield f, raw_display, tensor, warped_mask

    def release(self):
        pass


if __name__ == "__main__":
    p = prepare_buv_videos()
    stream = ClinicalUltrasoundStream(p)
    for i, raw, tensor in stream:
        if i >= 5:
            break
        print(f"Frame {i}: raw={raw.shape}, tensor={tensor.shape}, min={tensor.min():.2f}, max={tensor.max():.2f}")
    stream.release()
