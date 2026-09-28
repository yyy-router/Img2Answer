from pathlib import Path

import pymupdf


def create_sample_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=240, height=180)
    page.draw_rect(pymupdf.Rect(60, 50, 180, 130), color=(0, 0, 0), width=2)
    page.draw_line(pymupdf.Point(60, 50), pymupdf.Point(180, 130), color=(0, 0, 0), width=2)
    page.insert_text(pymupdf.Point(30, 30), "1. graphic sample", fontsize=10)
    document.new_page(width=240, height=180)
    document.save(path)
    document.close()

