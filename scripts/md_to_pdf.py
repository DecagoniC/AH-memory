"""Render Markdown documents to print-ready PDF via headless Chrome/Edge."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt;
  line-height: 1.45;
  color: #1d1d1f;
  margin: 0;
  text-align: justify;
  hyphens: auto;
}
h1, h2, h3, h4 { line-height: 1.25; letter-spacing: -0.01em; page-break-after: avoid; }
h1, h2, h3, h4 { text-align: left; }
h1 { font-size: 20pt; margin: 0 0 5pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; padding-bottom: 3pt; border-bottom: 1px solid #d2d2d7; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; }
h1 + h2 { margin-top: 8pt; border-bottom: 0; color: #424245; font-weight: 500; }
p, ul, ol { margin: 0 0 7pt; }
li { margin-bottom: 2pt; }
strong { font-weight: 600; }
hr { border: 0; border-top: 1px solid #d2d2d7; margin: 11pt 0; }
a { color: #0071e3; text-decoration: none; }
code {
  font-family: Consolas, "SF Mono", monospace;
  font-size: 9pt;
  background: #f5f5f7;
  border: 1px solid #e5e5ea;
  border-radius: 3px;
  padding: 0.5pt 3pt;
}
pre {
  background: #f5f5f7;
  border: 1px solid #e5e5ea;
  border-radius: 5px;
  padding: 8pt 10pt;
  margin: 0 0 10pt;
  overflow-wrap: break-word;
  white-space: pre-wrap;
  page-break-inside: avoid;
}
pre code { background: none; border: 0; padding: 0; font-size: 8.5pt; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0 0 12pt;
  font-size: 9pt;
  page-break-inside: avoid;
}
th, td { border: 1px solid #d2d2d7; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #f5f5f7; font-weight: 600; }
blockquote { margin: 0 0 10pt; padding-left: 10pt; border-left: 3px solid #d2d2d7; color: #6e6e73; }
em { color: #6e6e73; }
"""

BROWSERS = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)


def find_browser() -> Path:
    for name in ("chrome", "chromium", "google-chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for candidate in BROWSERS:
        if candidate.exists():
            return candidate
    raise SystemExit("Chrome/Edge not found: install one or pass --browser")


def render_html(markdown_path: Path) -> str:
    md = MarkdownIt("commonmark", {"html": True, "linkify": False}).enable("table")
    title = markdown_path.stem
    body = md.render(markdown_path.read_text(encoding="utf-8"))
    return (
        '<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


def to_pdf(markdown_path: Path, pdf_path: Path, browser: Path) -> None:
    html = render_html(markdown_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / f"{markdown_path.stem}.html"
        source.write_text(html, encoding="utf-8")
        result = subprocess.run(
            [
                str(browser),
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={tmp_dir / 'profile'}",
                f"--print-to-pdf={pdf_path.resolve()}",
                source.as_uri(),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    if not pdf_path.exists():
        raise SystemExit(
            f"PDF was not produced for {markdown_path}:\n{result.stderr.strip()}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Markdown files")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--browser", type=Path, default=None)
    parser.add_argument("--keep-html", action="store_true")
    args = parser.parse_args(argv)

    browser = args.browser or find_browser()
    for source in args.sources:
        if not source.exists():
            raise SystemExit(f"missing source: {source}")
        outdir = args.outdir or source.parent
        outdir.mkdir(parents=True, exist_ok=True)
        pdf_path = (outdir / source.name).with_suffix(".pdf")
        if args.keep_html:
            (outdir / source.name).with_suffix(".html").write_text(
                render_html(source), encoding="utf-8"
            )
        to_pdf(source, pdf_path, browser)
        print(f"{source} -> {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
