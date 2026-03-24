"""
Code execution utilities for SpecRepro.

Provides:
  - execute_code()  : run a Python snippet in a subprocess, capture stdout/stderr
  - extract_python_code() : pull ```python ... ``` blocks from LLM text
  - edit_code_lines()     : replace lines N–M in a code string
  - number_code_lines()   : add "L001: " prefixes for LLM-friendly diffs
"""

import subprocess
import sys
import re
import tempfile
import os
from typing import Optional


def execute_code(
    code: str,
    timeout: int = 120,
    working_dir: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Execute a Python code string in a subprocess.

    Returns:
        (stdout, error_message)  — error_message is None on success.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir,
        )
        stdout = result.stdout
        stderr = result.stderr if result.returncode != 0 else None
        return stdout, stderr
    except subprocess.TimeoutExpired:
        return "", f"TimeoutError: code exceeded {timeout}s"
    except Exception as e:
        return "", str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def extract_python_code(text: str) -> str:
    """
    Extract the first ```python ... ``` block from LLM output.

    Handles:
      - Normal: ```python\\n<code>\\n```
      - Truncated response: ```python\\n<code>  (no closing fence)
      - Raw text with no fences at all
    """
    text = text.strip()

    match = re.search(r"```python\s*\n(.*?)(?:^```|```\s*$)", text, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()

    match = re.search(r"```python\s*\n(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)(?:^```|```\s*$)", text, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()

    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_all_python_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` blocks from LLM output."""
    return [m.strip() for m in re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)]


def number_code_lines(code: str) -> str:
    """
    Return a version of the code with 'L001: ' prefixes on each line.
    Makes it easy for the LLM to refer to specific line numbers when editing.
    """
    lines = code.split("\n")
    return "\n".join(f"L{i+1:03d}: {line}" for i, line in enumerate(lines))


def edit_code_lines(code: str, start: int, end: int, new_lines: list[str]) -> str:
    """
    Replace lines [start, end] (1-indexed, inclusive) with new_lines.
    Used by CodeAgent to apply EDIT commands from the LLM.
    """
    lines = code.split("\n")
    # Convert to 0-indexed
    s, e = start - 1, end 
    s = max(0, min(s, len(lines)))
    e = max(s, min(e, len(lines)))
    updated = lines[:s] + new_lines + lines[e:]
    return "\n".join(updated)


def save_code(directory: str, filename: str, code: str):
    """Save code to a file, creating the directory if necessary."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path
