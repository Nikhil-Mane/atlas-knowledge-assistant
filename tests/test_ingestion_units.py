"""Detection, routing, parsers and chunk IDs. No services or API key needed."""
import shutil
from pathlib import Path

import pytest

from rag.config import Settings
from rag.ingestion.chunking import chunk_documents
from rag.ingestion.detect import detect
from rag.ingestion.parsers import ParseContext, get_parser, route
from rag.ingestion.schema import FileType, SourceInfo, assign_chunk_ids, make_doc

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def src_for(path: Path, det) -> SourceInfo:
    return SourceInfo(tenant="t", source_uri=path.as_uri(), filename=path.name,
                      sha256="0" * 64, mime=det.mime, file_type=det.file_type)


def ctx() -> ParseContext:
    return ParseContext(settings=Settings(_env_file=None))


@pytest.mark.parametrize("name, file_type, parser", [
    ("01-event-loop.md", FileType.markdown, "web_markup"),
    ("02-streams.html", FileType.html, "web_markup"),
    ("03-npm-packages.csv", FileType.csv, "tabular"),
    ("04-error-codes.json", FileType.json, "code_json"),
    ("05-express-server.js", FileType.code, "code_json"),
    ("06-security-checklist.txt", FileType.text, "fallback"),
    ("07-incident-email.eml", FileType.email, "email"),
    ("08-modules-guide.pdf", FileType.pdf, "pdf_text"),
    ("09-buffers-handwritten-scan.pdf", FileType.pdf, "ocr_vision"),
    ("10-cluster-whiteboard.png", FileType.image, "ocr_vision"),
    ("11-error-handling.docx", FileType.docx, "office"),
    ("12-async-patterns.pptx", FileType.pptx, "office"),
    ("13-lts-schedule.xlsx", FileType.xlsx, "tabular"),
])
def test_every_sample_is_detected_and_routed(name, file_type, parser):
    det = detect(SAMPLES / name)
    assert det.file_type is file_type
    assert route(det) == parser


def test_scanned_pdf_pages_are_found():
    assert detect(SAMPLES / "09-buffers-handwritten-scan.pdf").scanned_pages == [1]
    assert detect(SAMPLES / "08-modules-guide.pdf").scanned_pages == []


def test_content_beats_extension(tmp_path):
    fake = tmp_path / "report.txt"
    fake.write_text('{"code": "EMFILE", "fix": "use stream.pipeline()"}')
    assert detect(fake).file_type is FileType.json
    renamed = tmp_path / "scan.dat"
    shutil.copy(SAMPLES / "08-modules-guide.pdf", renamed)
    assert detect(renamed).file_type is FileType.pdf


def parse(name):
    path = SAMPLES / name
    det = detect(path)
    return get_parser(route(det))(path, det, src_for(path, det), ctx())


def test_csv_chunks_repeat_the_header():
    docs = parse("03-npm-packages.csv")
    assert docs and all("| package | category |" in d.page_content for d in docs)
    assert any("argon2" in d.page_content for d in docs)


def test_xlsx_keeps_title_and_reads_every_sheet():
    docs = parse("13-lts-schedule.xlsx")
    sheets = {d.metadata["sheet"] for d in docs}
    assert sheets == {"LTS schedule", "Upgrade notes"}
    lts = next(d for d in docs if d.metadata["sheet"] == "LTS schedule").page_content
    assert "Node.js release schedule" in lts            # title row above the header
    assert "| 22 | Jod | 2024-04-24 | 2024-10-29 | 2027-04-30 |" in lts


def test_json_keeps_code_and_fix_together():
    docs = parse("04-error-codes.json")
    emfile = [d for d in docs if "EMFILE" in d.page_content]
    assert emfile and "stream.pipeline()" in emfile[0].page_content


def test_code_is_split_with_filename_header():
    docs = parse("05-express-server.js")
    assert all(d.page_content.startswith("File: 05-express-server.js") for d in docs)
    assert any("SIGTERM" in d.page_content for d in docs)


def test_email_reads_body_headers_and_attachment():
    docs = parse("07-incident-email.eml")
    body, *attachments = docs
    assert body.metadata["email_subject"].startswith("Incident summary")
    assert "MaxListenersExceededWarning" in body.page_content
    assert attachments and attachments[0].metadata["attachment_name"] == "postmortem.txt"
    assert "40 minutes" in attachments[0].page_content
    # attachment chunks belong to the parent email for dedup and deletes
    assert attachments[0].metadata["doc_id"] == body.metadata["doc_id"]


def test_chunk_ids_are_stable_and_unique():
    src = SourceInfo("t", "file:///a.txt", "a.txt", "0" * 64, "text/plain", FileType.text)
    docs = lambda: [make_doc(t, src, parser="fallback", prechunked=True)
                    for t in ["alpha text", "beta text", "alpha text"]]
    first, second = assign_chunk_ids(docs()), assign_chunk_ids(docs())
    assert first == second                    # same content -> same IDs on re-ingest
    assert len(set(first)) == 3               # repeated text still gets its own ID


def test_free_text_is_split_with_context_header():
    src = SourceInfo("t", "file:///n.txt", "n.txt", "0" * 64, "text/plain", FileType.text)
    long_text = "\n\n".join(f"Paragraph {i}: " + "event loop " * 60 for i in range(20))
    chunks = chunk_documents([make_doc(long_text, src, parser="fallback")])
    assert len(chunks) > 3
    assert all(c.page_content.startswith("File: n.txt") for c in chunks)


def test_fast_pdf_parser_reads_text_per_page():
    path = SAMPLES / "08-modules-guide.pdf"
    det = detect(path)
    c = ParseContext(settings=Settings(_env_file=None, pdf_parser="fast"))
    docs = get_parser("pdf_text")(path, det, src_for(path, det), c)
    assert docs and all(d.metadata["parser"] == "pdf_fast" for d in docs)
    assert "import.meta.dirname" in " ".join(d.page_content for d in docs)
    assert chunk_documents(docs)[0].page_content.startswith("File: 08-modules-guide.pdf")
