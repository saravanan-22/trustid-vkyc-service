"""
Downloads the pretrained face models from the OpenCV Model Zoo and verifies
their SHA-256 checksums.

    python scripts/download_models.py [--dest models]

* YuNet face detector  (MIT licence)        ~0.2 MB
* SFace face recognizer (Apache-2.0 licence) ~37 MB
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "models"))
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for name, (url, expected) in MODELS.items():
        target = dest / name
        if target.is_file() and sha256(target) == expected:
            print(f"[ok]   {name} already present")
            continue
        print(f"[get]  {name}")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            print(f"[fail] {name}: checksum mismatch ({actual})", file=sys.stderr)
            return 1
        tmp.replace(target)
        print(f"[ok]   {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
