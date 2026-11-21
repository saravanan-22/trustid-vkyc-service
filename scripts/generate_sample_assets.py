"""
Regenerates the demo images in sample_assets/ and bruno/samples/.

The ID card is an obvious SPECIMEN: fictional name and details, a
checksum-valid but made-up Aadhaar number, and a public-domain NASA portrait
(Sally Ride, 1979). The "live" image is a different public-domain photo of the
same person (1984), so the 1:1 face match passes.

Usage:  python scripts/generate_sample_assets.py
"""

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FACES = ROOT / "tests" / "fixtures" / "faces"
OUT_DIRS = [ROOT / "sample_assets", ROOT / "bruno" / "samples"]

# Text a client-side OCR step would extract from the card (used by the collections).
CARD_TEXT = (
    "SPECIMEN - GOVERNMENT OF INDIA - AADHAAR\n"
    "Name: Demo Applicant\n"
    "DOB: 15/08/1994\n"
    "Gender: FEMALE\n"
    "9841 2341 5679"
)


def font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def make_id_card() -> Image.Image:
    card = Image.new("RGB", (600, 380), (245, 245, 245))
    draw = ImageDraw.Draw(card)
    draw.rectangle([0, 0, 600, 46], fill=(2, 132, 199))
    draw.text((300, 23), "GOVERNMENT OF INDIA - AADHAAR", font=font(22), fill="white", anchor="mm")

    portrait = Image.open(FACES / "ride_1979.jpg").convert("RGB")
    w, h = portrait.size
    side = int(min(w, h) * 0.62)
    cx, cy = w // 2, int(h * 0.42)
    portrait = portrait.crop((cx - side // 2, cy - int(side * 0.6), cx + side // 2, cy + int(side * 0.65)))
    portrait = portrait.resize((150, 186))
    card.paste(portrait, (30, 76))

    rows = [("Name: Demo Applicant", 108), ("DOB: 15/08/1994", 148), ("Gender: FEMALE", 188)]
    for text, y in rows:
        draw.text((205, y), text, font=font(18), fill=(30, 41, 59))
    draw.text((205, 238), "9841 2341 5679", font=font(28), fill=(15, 23, 42))

    # Diagonal SPECIMEN watermark.
    mark = Image.new("RGBA", card.size, (0, 0, 0, 0))
    ImageDraw.Draw(mark).text((300, 200), "SPECIMEN", font=font(84), fill=(220, 38, 38, 70), anchor="mm")
    mark = mark.rotate(18, resample=Image.BICUBIC, center=(300, 200))
    card = Image.alpha_composite(card.convert("RGBA"), mark).convert("RGB")

    draw = ImageDraw.Draw(card)
    draw.text((300, 352), "SPECIMEN - NOT A REAL DOCUMENT - SAMPLE DATA ONLY", font=font(13), fill=(185, 28, 28), anchor="mm")
    return card


def main() -> None:
    card = make_id_card()
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        card.save(out / "sample_id_card.jpg", quality=92)
        shutil.copyfile(FACES / "ride_1984.jpg", out / "sample_live_face.jpg")
        print(f"wrote {out.relative_to(ROOT)}/sample_id_card.jpg, sample_live_face.jpg")


if __name__ == "__main__":
    main()
