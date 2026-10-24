"""Render a cited PDF page with an exact-text highlight when PDF text permits it."""

from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def render_pdf_page(pdf_bytes: bytes, page_number: int, quote: str = "") -> bytes:
    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        if page_number < 1 or page_number > len(pdf):
            raise IndexError("Page not found")
        page = pdf[page_number - 1]
        width, height = page.get_size()
        scale = min(1.5, 1800 / max(width, height))
        image = page.render(scale=scale).to_pil().convert("RGBA")

        cleaned_quote = " ".join(quote.split())[:240]
        if len(cleaned_quote) >= 12:
            text_page = page.get_textpage()
            search_terms = [cleaned_quote, " ".join(cleaned_quote.split()[:6])]
            for term in dict.fromkeys(search_terms):
                if len(term) < 12:
                    continue
                match = text_page.search(term).get_next()
                if match is None:
                    continue
                start, count = match
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for index in range(text_page.count_rects(start, count)):
                    left, bottom, right, top = text_page.get_rect(index)
                    draw.rectangle(
                        (left * scale, (height - top) * scale,
                         right * scale, (height - bottom) * scale),
                        fill=(250, 204, 21, 105)
                    )
                image = Image.alpha_composite(image, overlay)
                break

        output = BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue()
    finally:
        pdf.close()
