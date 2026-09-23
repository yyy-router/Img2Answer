# Img2Answer

Img2Answer is a local-first Python project for preparing PDF question-bank materials for structured extraction and future image-based retrieval.

The repository is intentionally initialized with a small public surface. Local source PDFs, generated images, local design notes, vector stores, databases, and conda environments are not committed.

## Project Direction

The project aims to support:

- local PDF section processing
- target question-type extraction
- page rendering and image crop preparation
- structured question metadata storage
- future image-to-question retrieval

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

## Runtime Environment

The runtime environment should be created in the project root:

```powershell
conda env create -p .\.conda -f environment.yml
conda activate .\.conda
```

The `.conda/` directory is ignored by Git.

