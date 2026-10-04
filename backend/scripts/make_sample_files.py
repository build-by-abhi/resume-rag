"""
Generate a sample PDF + DOCX resume.

Real uploads matter: the TXT samples never exercise `pypdf`/`pdfminer` or
`python-docx`, so a broken PDF path would not show up until someone uploads an
actual resume. Run this once, or after changing the loaders:

    python scripts/make_sample_files.py

The PDF is written by hand (no extra dependency) so the repo does not need
reportlab just to produce a fixture.
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "sample_resumes"

PDF_LINES = [
    "Elena Vasquez",
    "Munich, Germany",
    "elena.vasquez@example.com | +49 89 5550 1122",
    "linkedin.com/in/elenavasquez",
    "",
    "SUMMARY",
    "Platform engineer with 9 years running container platforms and",
    "developer tooling for regulated enterprise environments.",
    "",
    "EXPERIENCE",
    "Principal Platform Engineer | Deutsche Telekom | 2020 - Present",
    "Migrated 300 services to Kubernetes across two regions.",
    "Built a GitOps delivery pipeline with ArgoCD and Terraform.",
    "Introduced OpenTelemetry tracing, cutting incident time by half.",
    "",
    "EDUCATION",
    "B.Sc. Informatics | Technical University of Munich | 2014",
    "",
    "SKILLS",
    "Cloud: AWS, Azure, Terraform, Kubernetes, Docker",
    "Languages: Go, Python, Bash",
    "Tooling: Prometheus, Grafana, Ansible, Jenkins",
]

DOCX_LINES = [
    "Sam Okafor",
    "Lisbon, Portugal",
    "sam.okafor@example.com | +351 21 555 8899",
    "github.com/samokafor",
    "",
    "SUMMARY",
    "Backend engineer focused on JVM systems, event-driven architecture and",
    "developer experience. Recently moved from fintech to climate tech.",
    "",
    "EXPERIENCE",
    "Senior Software Engineer | Klarna | 2021 - Present",
    "Designed a Kotlin and Spring Boot service handling 12k requests per second.",
    "Replaced a nightly batch with a Kafka streaming pipeline, cutting latency",
    "from 6 hours to under a minute.",
    "Migrated the primary datastore from MySQL to PostgreSQL online.",
    "",
    "EDUCATION",
    "M.Sc. Computer Science | University of Lisbon | 2018",
    "",
    "SKILLS",
    "Languages: Kotlin, Java, SQL, Python",
    "Frameworks: Spring Boot, Kafka, Gradle",
    "Data: PostgreSQL, Redis, Elasticsearch",
    "Cloud: GCP, Docker, Kubernetes, Terraform",
]


def write_pdf(lines: list[str], path: Path) -> None:
    """Write a single-page PDF with one text line per row, no dependencies."""

    def escape(value: str) -> str:
        return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    content = ["BT", "/F1 11 Tf", "14 TL", "1 0 0 1 60 780 Tm"]
    content += [f"({escape(line)}) Tj T*" for line in lines]
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))


def write_docx(lines: list[str], path: Path) -> None:
    import docx

    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    document.save(str(path))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "elena_platform_engineer.pdf"
    docx_path = OUT_DIR / "sam_kotlin_backend.docx"

    write_pdf(PDF_LINES, pdf_path)
    write_docx(DOCX_LINES, docx_path)

    print(f"wrote {pdf_path.name} ({pdf_path.stat().st_size} bytes)")
    print(f"wrote {docx_path.name} ({docx_path.stat().st_size} bytes)")
    print("\nRe-ingest them with:  python -m scripts.seed --reset")


if __name__ == "__main__":
    main()