import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from onnxruntime.quantization import CalibrationDataReader

from src.utils.seed import seed_everything


class BUSICalibrationDataReader(CalibrationDataReader):
    """
    Feeds calibration samples from the training set using exact inference preprocessing:
    - 256x256 grayscale replicated to 3 channels
    - Standard ImageNet mean/std normalization
    - Shape: (1, 3, 256, 256), dtype: float32
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        num_images: int = 100,
        seed: int = 42,
        splits_file: str = "data/splits.json",
        data_dir: str = "data/processed",
        input_name: str = "input",
        image_ids: Optional[List[str]] = None,
    ):
        super().__init__()
        self.input_name = input_name
        self.images_dir = Path(data_dir) / "images"

        if image_ids is not None:
            self.selected_ids = list(image_ids)
        else:
            with open(splits_file, "r") as f:
                splits = json.load(f)
            train_ids = splits["train"]

            seed_everything(seed)
            rng = np.random.RandomState(seed)
            n = min(num_images, len(train_ids))
            self.selected_ids = list(rng.choice(train_ids, size=n, replace=False))

        self.enum_data = iter(self.selected_ids)

    def get_next(self) -> Optional[dict]:
        try:
            img_id = next(self.enum_data)
        except StopIteration:
            return None

        img_path = self.images_dir / f"{img_id}.png"
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Missing calibration image: {img_path}")

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        img_transposed = img_norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 256, 256)

        return {self.input_name: img_transposed}

    def rewind(self):
        self.enum_data = iter(self.selected_ids)
