import struct
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

URL = "https://huggingface.co/datasets/huihuixu/uterine_fibroid_ultrasound_video_segmentation/resolve/main/UFUV.zip"
CD_OFFSET = 1586327139
CD_SIZE = 1093571

VIDEOS = ["IM_0003", "IM_0004", "IM_0005", "IM_0014", "IM_0055"]


def fetch_file(entry: Tuple[str, int, int, int], base_dir: Path) -> str:
    fname, local_offset, comp_size, uncomp_size = entry
    # Read local header (up to 150 bytes) + compressed data
    req = urllib.request.Request(URL, headers={"Range": f"bytes={local_offset}-{local_offset + comp_size + 150}"})
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()

    loc_fn_len = struct.unpack("<H", raw[26:28])[0]
    loc_ext_len = struct.unpack("<H", raw[28:30])[0]
    start = 30 + loc_fn_len + loc_ext_len
    comp_data = raw[start : start + comp_size]
    decomp = zlib.decompress(comp_data, -15)

    # Save to disk
    # fname is like 'uterine/test/images/IM_0003/00013.png'
    parts = fname.split("/")
    img_or_mask = parts[2]  # 'images' or 'masks'
    vid_id = parts[3]        # 'IM_0003'
    frame_name = parts[4]    # '00013.png'

    target_path = base_dir / vid_id / img_or_mask / frame_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(decomp)

    return str(target_path)


def download_ufuv_5_test_videos(out_dir: str = "data/ufuv_test"):
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching Central Directory from UFUV.zip...")
    req = urllib.request.Request(URL, headers={"Range": f"bytes={CD_OFFSET}-{CD_OFFSET + CD_SIZE - 1}"})
    with urllib.request.urlopen(req) as resp:
        cd_data = resp.read()

    entries = []
    idx = 0
    while idx < len(cd_data):
        if cd_data[idx : idx + 4] != b"PK\x01\x02":
            break
        fname_len = struct.unpack("<H", cd_data[idx + 28 : idx + 30])[0]
        extra_len = struct.unpack("<H", cd_data[idx + 30 : idx + 32])[0]
        comm_len = struct.unpack("<H", cd_data[idx + 32 : idx + 34])[0]
        local_offset = struct.unpack("<I", cd_data[idx + 42 : idx + 46])[0]
        comp_size = struct.unpack("<I", cd_data[idx + 20 : idx + 24])[0]
        uncomp_size = struct.unpack("<I", cd_data[idx + 24 : idx + 28])[0]
        fname = cd_data[idx + 46 : idx + 46 + fname_len].decode("utf-8", errors="ignore")

        if any(f"/{v}/" in fname for v in VIDEOS) and fname.endswith(".png"):
            entries.append((fname, local_offset, comp_size, uncomp_size))
        idx += 46 + fname_len + extra_len + comm_len

    print(f"Found {len(entries)} PNG files across 5 real clinical test videos ({VIDEOS}).")
    t0 = time.perf_counter()

    # Parallel download with 20 threads
    with ThreadPoolExecutor(max_workers=20) as pool:
        saved_paths = list(pool.map(lambda e: fetch_file(e, base_dir), entries))

    dt = time.perf_counter() - t0
    print(f"Successfully downloaded and extracted {len(saved_paths)} genuine video frames & masks in {dt:.2f}s ({len(saved_paths)/dt:.1f} files/s)!")


if __name__ == "__main__":
    download_ufuv_5_test_videos()
