# Test fixtures

## `dummy.pdf`

- **Source:** https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
- **Provenance:** W3C Web Accessibility Initiative test files.
- **Content:** single page, PDF 1.4, deflate-encoded, contains the text
  "Dummy PDF file".
- **Why this file:** smallest publicly-served PDF with normal text that
  validates the real `pdf-extract` → `PdfExtractCrate::extract` round
  trip in `tests/ingest_pdf_e2e.rs` and the extractor unit tests.
