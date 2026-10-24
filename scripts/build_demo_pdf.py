"""Rebuild the small public-domain sample PDF used by the source viewer demo."""

from io import BytesIO
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


LINES = [
    "InsightPilot Source Viewer Demo",
    "The evidence viewer opens the original PDF page and highlights the exact cited passage.",
    "Each indexed excerpt carries a document ID, chunk ID, and page number.",
    "The public demo keeps only bundled sample documents in memory.",
]


def build_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    commands = ["BT", "/F1 14 Tf", "50 735 Td"]
    for line in LINES:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.extend([f"({escaped}) Tj", "0 -30 Td"])
    commands.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "demo_data" / "source_viewer_demo.pdf"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(build_pdf())
    print(f"Wrote {target}")
