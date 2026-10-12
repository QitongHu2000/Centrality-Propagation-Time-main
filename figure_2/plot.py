"""Reuse the submitted structural artwork and correct its CC heading typo."""

import pymupdf

from build_record import prepare, record
from paths import ASSETS, figure_path


def build():
    source = ASSETS / "original.pdf"
    target = figure_path()
    with pymupdf.open(source) as doc:
        page = doc[0]
        headings = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"] == "Clossness Centrality"
        ]
        assert len(headings) == 1
        heading = headings[0]
        font_entry = next(
            entry
            for entry in page.get_fonts()
            if entry[3].split("+")[-1] == heading["font"]
        )
        font_data = doc.extract_font(font_entry[0])[3]
        font = pymupdf.Font(fontbuffer=font_data)
        title = "Closeness Centrality"
        width = font.text_length(title, fontsize=heading["size"])
        bounds = pymupdf.Rect(heading["bbox"])
        origin = ((bounds.x0 + bounds.x1 - width) / 2, heading["origin"][1])
        page.add_redact_annot(bounds, fill=(1, 1, 1))
        # Preserve every graph image and vector drawing.
        page.apply_redactions(images=0, graphics=0)
        page.insert_font(fontname="corrected_heading", fontbuffer=font_data)
        page.insert_text(
            origin,
            title,
            fontname="corrected_heading",
            fontsize=heading["size"],
            color=(0, 0, 0),
        )
        doc.save(target, garbage=3, deflate=True)
        page.get_pixmap(matrix=pymupdf.Matrix(0.5, 0.5)).save(
            target.with_suffix(".png")
        )
    return {source}


if __name__ == "__main__":
    prepare()
    record(build())
