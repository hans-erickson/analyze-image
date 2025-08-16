#!/usr/bin/env python3
r"""
Analyze one or many images with a text prompt using OpenAI's API and emit LaTeX,
optionally compiling directly to PDF.

Improvements:
- Safer LaTeX extraction (handles <latex>…</latex>, ```latex fences, $$…$$, \[ … \], inline $…$)
- CLI: --emit {tex,pdf,both}, --engine {auto,tectonic,pdflatex,xelatex,lualatex}
- Robust PDF compilation with subprocess + helpful diagnostics
"""

from __future__ import annotations

import argparse
import base64
import csv
import logging
import os
import re
import sys
import tempfile
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional

from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError, APIError


# ----------------------------- Data structures -----------------------------

@dataclass
class WorkItem:
    image_file: str
    output_file: str
    prompt_file: Optional[str] = None  # optional per-item override


@dataclass
class EmitOptions:
    emit: str                  # "tex", "pdf", "both"
    engine: str                # "auto", "tectonic", "pdflatex", "xelatex", "lualatex"
    document_class: str        # for --latex-output=document support if desired later
    latex_mode: str            # "fragment" or "document"
    max_tokens: int
    overwrite: bool


# ----------------------------- IO utilities --------------------------------

def read_text(path: str, desc: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{desc} not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to read {desc}: {path}") from e


def read_api_key(path: str) -> str:
    key = read_text(path, "API key file").strip()
    if not key:
        raise ValueError("API key file is empty.")
    return key


def image_to_data_url(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".tiff": "image/tiff",
    }.get(ext, "application/octet-stream")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------- LaTeX post-processing --------------------------
# Preferred containers first
_LATEX_TAG_RE = re.compile(r"<latex>(.*?)</latex>", re.DOTALL | re.IGNORECASE)
_FENCE_LATEX_RE = re.compile(r"```(?:latex)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FENCE_ANY_RE = re.compile(r"```+\s*(.*?)\s*```+", re.DOTALL)  # fallback if unlabeled

# Display math
_DOLLAR_DOLLAR_RE = re.compile(r"(?s)(?<!\\)\$\$(.+?)(?<!\\)\$\$")
_BRACKET_DISPLAY_RE = re.compile(r"(?s)\\\[(.+?)\\\]")

# Inline math (avoid $$ and respect escapes)
_INLINE_DOLLAR_RE = re.compile(r"(?s)(?<!\\)\$(?!\$)\s*(.+?)\s*(?<!\\)\$(?!\$)")

def extract_latex(text: str) -> str:
    r"""
    Extract LaTeX robustly with a prioritized strategy:
      1) <latex>...</latex>
      2) ```latex ... ```
      3) $$ ... $$
      4) \[ ... \]
      5) inline $ ... $
      6) unlabeled ``` ... ``` (last resort)
      7) otherwise: return the whole text
    """
    if not text:
        return ""

    m = _LATEX_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _FENCE_LATEX_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _DOLLAR_DOLLAR_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _BRACKET_DISPLAY_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _INLINE_DOLLAR_RE.search(text)
    if m:
        return m.group(1).strip()

    m = _FENCE_ANY_RE.search(text)
    if m:
        return m.group(1).strip()

    return text.strip()


def wrap_as_document(body: str, document_class: str = "article") -> str:
    preamble = rf"""\documentclass{{{document_class}}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\begin{document}
"""
    return f"{preamble}\n{body}\n\\end{document}\n"


# ----------------------------- OpenAI call ---------------------------------

_FORMAT_SYSTEM_INSTRUCTION = (
    "You are an expert that analyzes images and returns LaTeX ONLY. "
    "Wrap ALL LaTeX content inside a single <latex>...</latex> block. "
    "Do not include Markdown fences, commentary, or any text outside <latex>…</latex>."
)

def analyze_image_with_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    image_path: str,
    max_tokens: int = 800,
) -> str:
    data_url = image_to_data_url(image_path)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FORMAT_SYSTEM_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            max_tokens=max_tokens,
        )
    except (APIConnectionError, RateLimitError, APIStatusError, APIError) as e:
        raise

    return (resp.choices[0].message.content or "").strip()


# ----------------------------- PDF compilation -----------------------------

def find_engine(preference: list[str]) -> Optional[str]:
    for exe in preference:
        if shutil.which(exe):
            return exe
    return None


def compile_pdf(tex_source: str, out_pdf: str, engine: str) -> None:
    """
    Compile LaTeX source string to a PDF at out_pdf using the chosen engine.
    Uses a temp build directory; writes a .log next to the PDF if compilation fails.
    """
    base = os.path.splitext(os.path.basename(out_pdf))[0]
    with tempfile.TemporaryDirectory(prefix="latex_build_") as work:
        tex_path = os.path.join(work, base + ".tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_source)

        cmd: list[str]
        if engine == "tectonic":
            # Tectonic is single-pass and fetches packages automatically
            cmd = ["tectonic", "--keep-logs", "--synctex", "-o", work, tex_path]
        elif engine in ("pdflatex", "xelatex", "lualatex"):
            # Run twice to resolve refs; quiet-ish output
            cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", tex_path]
        else:
            raise ValueError(f"Unknown LaTeX engine: {engine}")

        def run_cmd(c: list[str]) -> None:
            proc = subprocess.run(c, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"{c[0]} failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

        if engine == "tectonic":
            run_cmd(cmd)
        else:
            # two passes for TeX engines
            run_cmd(cmd)
            run_cmd(cmd)

        built_pdf = os.path.join(work, base + ".pdf")
        if not os.path.isfile(built_pdf):
            raise RuntimeError("LaTeX engine reported success but no PDF was produced.")

        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        shutil.move(built_pdf, out_pdf)


# ----------------------------- Batch helpers -------------------------------

def choose_pdf_path(target_path: str) -> str:
    """If the provided path ends with .pdf, use it; otherwise replace extension with .pdf."""
    root, ext = os.path.splitext(target_path)
    return target_path if ext.lower() == ".pdf" else root + ".pdf"


def ensure_tex_path(target_path: str) -> str:
    """If the provided path ends with .tex, use it; otherwise replace extension with .tex."""
    root, ext = os.path.splitext(target_path)
    return target_path if ext.lower() == ".tex" else root + ".tex"


def process_item(
    client: OpenAI,
    model: str,
    default_prompt: str,
    item: WorkItem,
    opts: EmitOptions,
) -> tuple[WorkItem, dict, Optional[Exception]]:
    """
    Returns (item, outputs, err) where outputs includes any written paths:
      {'tex': '/path/file.tex' or None, 'pdf': '/path/file.pdf' or None}
    """
    try:
        # Determine desired outputs based on CLI and extension
        want_tex = opts.emit in ("tex", "both")
        want_pdf = opts.emit in ("pdf", "both")

        # If user gave .pdf in output and emit=auto, honor that; here we keep it simple:
        tex_path = ensure_tex_path(item.output_file) if want_tex or (want_pdf and not want_tex) else None
        pdf_path = choose_pdf_path(item.output_file) if want_pdf else None

        # Overwrite guard
        if not opts.overwrite:
            if tex_path and os.path.exists(tex_path):
                return item, {}, FileExistsError(f"Output exists (use --overwrite): {tex_path}")
            if pdf_path and os.path.exists(pdf_path):
                return item, {}, FileExistsError(f"Output exists (use --overwrite): {pdf_path}")

        prompt = read_text(item.prompt_file, "Per-item prompt") if item.prompt_file else default_prompt
        raw = analyze_image_with_openai(client, model, prompt, item.image_file, max_tokens=opts.max_tokens)
        latex_body = extract_latex(raw)
        if opts.latex_mode == "document":
            latex_source = wrap_as_document(latex_body, document_class=opts.document_class)
        else:
            latex_source = latex_body

        written = {}

        if want_tex:
            os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_source)
            written["tex"] = tex_path

        if want_pdf:
            # If we didn't write a .tex file (emit=pdf only), compile directly from memory.
            engine_choice = resolve_engine(opts.engine)
            pdf_out = pdf_path
            compile_pdf(latex_source, pdf_out, engine_choice)
            written["pdf"] = pdf_out

        return item, written, None

    except Exception as e:
        return item, {}, e


def resolve_engine(engine_opt: str) -> str:
    if engine_opt != "auto":
        if not shutil.which(engine_opt):
            raise FileNotFoundError(
                f"LaTeX engine '{engine_opt}' not found in PATH. "
                "Install it or use --engine auto."
            )
        return engine_opt

    # auto: prefer tectonic → xelatex → lualatex → pdflatex
    found = find_engine(["tectonic", "xelatex", "lualatex", "pdflatex"])
    if not found:
        raise FileNotFoundError(
            "No LaTeX engine found (looked for: tectonic, xelatex, lualatex, pdflatex). "
            "Install one and re-run, or use --emit tex."
        )
    return found


def load_manifest_csv(
    manifest_path: str,
    require_columns: Iterable[str] = ("image", "output"),
) -> list[WorkItem]:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    items: list[WorkItem] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in require_columns if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
        for row in reader:
            img = (row.get("image") or "").strip()
            out = (row.get("output") or "").strip()
            pfile = (row.get("prompt_file") or "").strip() or None
            if not img or not out:
                logging.warning("Skipping row with empty image/output: %r", row)
                continue
            items.append(WorkItem(image_file=img, output_file=out, prompt_file=pfile))
    return items


# ----------------------------- CLI + main ----------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze image(s) with a text prompt using OpenAI and output LaTeX and/or PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Shared
    p.add_argument("prompt_file", help="Path to text file containing the default prompt")
    p.add_argument("api_key_file", help="Path to file containing the OpenAI API key")

    # Modes (mutually exclusive): single or batch manifest
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to a single input image (single mode)")
    mode.add_argument("--manifest", help="CSV manifest for batch mode")

    # Outputs for single mode
    p.add_argument("--output", help="Output base path for single mode (use .tex or .pdf)")

    # OpenAI options
    p.add_argument("--model", default="gpt-4o-mini", help="Vision-capable model (e.g., gpt-4o-mini, gpt-5-mini)")
    p.add_argument("--max-tokens", type=int, default=800, help="Max output tokens per analysis")

    # Emission/compilation
    p.add_argument("--emit", choices=["tex", "pdf", "both"], default="tex",
                   help="Write LaTeX (.tex), PDF (.pdf), or both")
    p.add_argument("--engine", choices=["auto", "tectonic", "pdflatex", "xelatex", "lualatex"],
                   default="auto", help="LaTeX engine to compile PDFs (auto tries several)")
    p.add_argument("--latex-mode", choices=["fragment", "document"], default="fragment",
                   help="Emit a raw LaTeX fragment or a full compilable document")
    p.add_argument("--document-class", default="article",
                   help="Document class when --latex-mode=document")

    # Misc
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    p.add_argument("--workers", type=int, default=4, help="Max parallel workers for batch mode")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate single-mode output arg
    if args.image and not args.output:
        logging.error("--output is required in single mode (use .tex or .pdf)")
        return 2

    try:
        default_prompt = read_text(args.prompt_file, "Prompt file")
        api_key = read_api_key(args.api_key_file)
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.error("Initialization failed: %s", e)
        return 1

    opts = EmitOptions(
        emit=args.emit,
        engine=args.engine,
        document_class=args.document_class,
        latex_mode=args.latex_mode,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
    )

    # ------------------- Single mode -------------------
    if args.image:
        item = WorkItem(image_file=args.image, output_file=args.output)
        _, outdict, err = process_item(
            client=client,
            model=args.model,
            default_prompt=default_prompt,
            item=item,
            opts=opts,
        )
        if err:
            logging.error("Failed: %s", err)
            return 3
        for k, v in outdict.items():
            logging.info("Wrote %s: %s", k, v)
        return 0

    # ------------------- Batch mode --------------------
    try:
        items = load_manifest_csv(args.manifest)
        if not items:
            logging.warning("Manifest has no valid rows.")
            return 0
    except Exception as e:
        logging.error("Could not load manifest: %s", e)
        return 4

    errors = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for it in items:
            futures.append(
                pool.submit(
                    process_item,
                    client,
                    args.model,
                    default_prompt,
                    it,
                    opts,
                )
            )
        for fut in as_completed(futures):
            item, outdict, err = fut.result()
            if err:
                errors += 1
                logging.error("FAILED [%s] -> %s (%s)", item.image_file, item.output_file, err)
            else:
                wrote = ", ".join(f"{k}:{v}" for k, v in outdict.items())
                logging.info("OK     [%s] -> %s", item.image_file, wrote)

    if errors:
        logging.warning("Completed with %d error(s).", errors)
        return 5
    logging.info("Batch complete. %d item(s) processed.", len(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
