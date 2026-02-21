#!/usr/bin/env python3
"""
Convert Python percent-format notebooks (.py with # %% markers) to .ipynb format.

Handles:
- Module docstrings before the first # %% marker -> markdown cell
- # %% [markdown] cells -> markdown (strips leading '# ' from lines)
- # %% or # %% Title cells -> code cells
- # !pip install lines -> code cells (shell commands)
"""

import json
import os
import re
import sys


def make_notebook(cells):
    """Create a valid nbformat 4 notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3"
            }
        },
        "cells": cells
    }


def make_cell(cell_type, source):
    """Create a notebook cell dict."""
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }


def strip_markdown_comments(lines):
    """
    Strip leading '# ' or '#' prefix from markdown cell lines.
    Returns list of strings (each ending with \n except possibly the last).
    """
    result = []
    for line in lines:
        if line.startswith("# "):
            result.append(line[2:])
        elif line.rstrip() == "#":
            # Bare '#' line -> empty line in markdown
            result.append("\n")
        else:
            # Lines that don't start with # (shouldn't happen in markdown cells,
            # but handle gracefully -- keep as-is)
            result.append(line)
    return result


def parse_docstring_block(lines):
    """
    If the file starts with a triple-quoted docstring, extract it as raw text
    (without the triple quotes). Returns (docstring_lines, remaining_lines) or
    (None, lines) if no docstring found.
    """
    if not lines:
        return None, lines

    first_line = lines[0].rstrip()
    if first_line.startswith('"""') or first_line.startswith("'''"):
        quote = first_line[:3]
        # Check for single-line docstring
        if first_line.endswith(quote) and len(first_line) > 3:
            inner = first_line[3:-3]
            return [inner + "\n"], lines[1:]

        # Multi-line docstring
        doc_lines = []
        # Content on the same line as opening quotes
        remainder = first_line[3:]
        if remainder:
            doc_lines.append(remainder + "\n")

        for i, line in enumerate(lines[1:], start=1):
            if quote in line:
                # Found closing quotes
                before_close = line[:line.index(quote)]
                if before_close.strip():
                    doc_lines.append(before_close.rstrip() + "\n")
                return doc_lines, lines[i + 1:]
            else:
                doc_lines.append(line)

    return None, lines


def ensure_newlines(lines):
    """Ensure all lines end with newline, except strip trailing blank lines."""
    if not lines:
        return []

    # Ensure each line ends with \n
    result = []
    for line in lines:
        if not line.endswith("\n"):
            result.append(line + "\n")
        else:
            result.append(line)

    # Strip trailing empty lines
    while result and result[-1].strip() == "":
        result.pop()

    # Remove trailing \n from last line (notebook convention)
    if result and result[-1].endswith("\n"):
        result[-1] = result[-1][:-1]

    return result


def is_pip_install_line(line):
    """Check if a line is a pip install command (commented shell command)."""
    stripped = line.strip()
    return stripped.startswith("# !pip") or stripped.startswith("#!pip")


def convert_py_to_ipynb(py_path):
    """Convert a percent-format .py file to .ipynb."""
    with open(py_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    cells = []

    # Phase 1: Extract leading docstring (before any # %% marker)
    docstring_lines, remaining_lines = parse_docstring_block(raw_lines)

    # Find where the first # %% marker is
    first_marker_idx = None
    for i, line in enumerate(remaining_lines):
        if line.rstrip().startswith("# %%"):
            first_marker_idx = i
            break

    # Any non-empty lines between docstring end and first marker?
    # (These would be stray code/comments -- skip blank lines)
    if first_marker_idx is not None:
        pre_marker = remaining_lines[:first_marker_idx]
    else:
        pre_marker = remaining_lines
        remaining_lines = []

    # Skip the docstring cell -- the first # %% [markdown] has the proper intro
    # (per user instructions). But if there's no markdown cell coming, use it.
    # We'll skip the docstring entirely since instructions say "skip it since the
    # first # %% [markdown] has the proper intro".

    # Phase 2: Split remaining lines into cell blocks at # %% markers
    if first_marker_idx is not None:
        lines_from_markers = remaining_lines[first_marker_idx:]
    else:
        lines_from_markers = []

    # Parse into raw blocks: each block = (marker_line, [content_lines])
    blocks = []
    current_marker = None
    current_lines = []

    for line in lines_from_markers:
        if line.rstrip().startswith("# %%"):
            if current_marker is not None:
                blocks.append((current_marker, current_lines))
            current_marker = line.rstrip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last block
    if current_marker is not None:
        blocks.append((current_marker, current_lines))

    # Phase 3: Convert blocks to cells
    for marker, content_lines in blocks:
        is_markdown = "[markdown]" in marker

        # Strip leading and trailing blank lines from content
        while content_lines and content_lines[0].strip() == "":
            content_lines.pop(0)
        while content_lines and content_lines[-1].strip() == "":
            content_lines.pop()

        if not content_lines:
            # Empty cell -- skip
            continue

        if is_markdown:
            # Convert markdown: strip leading '# ' from each line
            md_lines = strip_markdown_comments(content_lines)
            md_lines = ensure_newlines(md_lines)
            if md_lines:
                cells.append(make_cell("markdown", md_lines))
        else:
            # Code cell -- check if it's a pip install line
            # The marker might be "# %% Install dependencies" with content like
            # "# !pip install ..."
            # Check if ALL non-blank lines are pip install comments
            code_lines = content_lines[:]

            # Check for pip install lines that are commented out
            # These should become uncommented shell commands in code cells
            processed = []
            for cl in code_lines:
                if is_pip_install_line(cl):
                    # Strip the leading "# " to make it an actual !pip command
                    stripped = cl.strip()
                    if stripped.startswith("# !"):
                        processed.append(stripped[2:] + "\n")
                    elif stripped.startswith("#!"):
                        processed.append(stripped[1:] + "\n")
                    else:
                        processed.append(cl)
                else:
                    processed.append(cl)

            processed = ensure_newlines(processed)
            if processed:
                cells.append(make_cell("code", processed))

    # Write the notebook
    nb = make_notebook(cells)
    ipynb_path = os.path.splitext(py_path)[0] + ".ipynb"
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    return ipynb_path


def main():
    notebook_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")
    py_files = [
        os.path.join(notebook_dir, "engram_kaggle.py"),
        os.path.join(notebook_dir, "engram_lora_finetune.py"),
    ]

    for py_file in py_files:
        if not os.path.exists(py_file):
            print(f"ERROR: File not found: {py_file}")
            continue

        ipynb_path = convert_py_to_ipynb(py_file)
        print(f"Converted: {py_file}")
        print(f"  -> {ipynb_path}")

        # Validate: read back and parse JSON
        with open(ipynb_path, "r") as f:
            nb = json.load(f)

        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  Cells: {len(nb['cells'])} total ({n_code} code, {n_md} markdown)")
        print(f"  nbformat: {nb['nbformat']}.{nb['nbformat_minor']}")
        print()


if __name__ == "__main__":
    main()
