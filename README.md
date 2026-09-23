# Img2Answer

Img2Answer is a local-first Python project for preparing PDF question-bank materials for structured extraction and future image-based retrieval.

The current implementation is a first-stage POC. It focuses on reading configured PDF page ranges, rendering pages, generating simple graphic crop candidates, and writing a JSON report. Local source PDFs, generated images, local design notes, vector stores, databases, and conda environments are not committed.

## Current Features

- Read YAML or JSON section configuration.
- Inspect local PDF files and collect basic metadata:
  - file path
  - page count
  - SHA-256 hash
- Validate configured page ranges.
- Render configured PDF sections into PNG page images.
- Generate simple crop candidates from rendered pages.
- Write a JSON processing report.
- Provide a CLI entry point.
- Include unit and integration tests using generated sample PDFs, not real local materials.

## Not Implemented Yet

- OCR
- ChromaDB vector storage
- image embedding models
- text or image search API
- web UI
- full question parsing
- database-backed question storage

## Repository Rules

The following paths are local-only and ignored by Git:

```text
data/
.conda/
docs/
```

Do not commit:

- original PDFs
- generated page images or crop images
- local design documents
- local vector databases
- local SQLite or database files
- credentials, tokens, or account information

## Environment

Create the conda environment in the project root:

```powershell
conda env create -p .\.conda -f environment.yml
conda activate .\.conda
```

The environment is intentionally stored at:

```text
.\.conda
```

## Configuration

Use a section config to describe which PDF pages should be processed.

Example:

```yaml
documents:
  sample_doc:
    path: data/raw/sample.pdf
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 3
```

An example config is provided at:

```text
configs/sample.sections.example.yml
```

Page numbers in config files are human-facing and start at `1`.

## Run The POC

From the project root:

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m img2answer.cli --config configs\sample.sections.example.yml --output-dir data\processed --dpi 300
```

Outputs are written under:

```text
data/processed/
  pages/
  crops/
  reports/
```

These outputs are ignored by Git.

## Run Tests

```powershell
$env:PYTHONPATH = "src"
.\.conda\python.exe -m unittest discover -s tests -v
```

Current test coverage verifies:

- config loading
- invalid page range handling
- PDF metadata inspection
- section rendering
- crop candidate generation
- report writing

## Development Workflow

This project follows an issue-first and SDD-style workflow:

1. Create or draft an issue.
2. Write and review the local design/spec.
3. Implement on a feature branch.
4. Add tests with the implementation.
5. Open a PR and link the issue with `Closes #<issue-number>`.

Branch names should be short and do not need to contain the issue number:

```text
feature/pdf-section-crop-poc
fix/page-range-validation
```

