"""Figure 3: all eleven methods in the original epidemic score-scatter layout."""

from pathlib import Path

import pymupdf

from manuscript_assets import INPUTS, build_score_patterns
from paths import figure_path


from build_record import prepare, record


def build():
    build_score_patterns("E")
    # Reuse the submitted vector artwork, including its icon, equation and frame.
    card_path = Path(__file__).with_name("dynamics_card.pdf")
    INPUTS.add(card_path)
    target = figure_path("scores_E")
    with pymupdf.open(target) as doc, pymupdf.open(card_path) as card:
        page = doc[0]
        width = 0.42 * page.rect.width
        height = width * card[0].rect.height / card[0].rect.width
        left, top = (page.rect.width - width) / 2, 0.004 * page.rect.height
        page.show_pdf_page(pymupdf.Rect(left, top, left + width, top + height), card, 0)
        result = doc.tobytes(garbage=4, deflate=True)
    target.write_bytes(result)
    with pymupdf.open(target) as doc:
        doc[0].get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5)).save(
            target.with_suffix(".png")
        )


if __name__ == "__main__":
    prepare()
    build()
    record(INPUTS)
