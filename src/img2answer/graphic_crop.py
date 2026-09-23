from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops


@dataclass(frozen=True)
class CropCandidate:
    source_page: Path
    output_path: Path
    bbox: tuple[int, int, int, int]
    width: int
    height: int


def crop_graphic_candidates(
    page_image: str | Path,
    output_dir: str | Path,
    *,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.75,
    padding: int = 8,
) -> list[CropCandidate]:
    image_path = Path(page_image)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        bbox = _content_bbox(rgb, padding)
        if bbox is None:
            return []

        image_area = rgb.width * rgb.height
        crop_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if crop_area < image_area * min_area_ratio or crop_area > image_area * max_area_ratio:
            return []

        crop = rgb.crop(bbox)
        crop_path = output_path / f"{image_path.stem}-crop-001.png"
        crop.save(crop_path)
        return [
            CropCandidate(
                source_page=image_path,
                output_path=crop_path,
                bbox=bbox,
                width=crop.width,
                height=crop.height,
            )
        ]


def crop_pages(
    page_images: list[Path],
    output_dir: str | Path,
    *,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.75,
    padding: int = 8,
) -> list[CropCandidate]:
    candidates: list[CropCandidate] = []
    for page_image in page_images:
        candidates.extend(
            crop_graphic_candidates(
                page_image,
                output_dir,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                padding=padding,
            )
        )
    return candidates


def _content_bbox(image: Image.Image, padding: int) -> tuple[int, int, int, int] | None:
    grayscale = image.convert("L")
    mask = grayscale.point(lambda value: 255 if value < 245 else 0, mode="1")
    bbox = mask.getbbox()
    if bbox is None:
        return None

    # Trim a solid border when a rendered test or scanned page has one.
    crop = image.crop(bbox)
    border = Image.new("RGB", crop.size, "white")
    diff_bbox = ImageChops.difference(crop, border).getbbox()
    if diff_bbox is not None:
        bbox = (
            bbox[0] + diff_bbox[0],
            bbox[1] + diff_bbox[1],
            bbox[0] + diff_bbox[2],
            bbox[1] + diff_bbox[3],
        )

    return (
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(image.width, bbox[2] + padding),
        min(image.height, bbox[3] + padding),
    )

