"""Offline task checks. HumanEval execution fails closed without bubblewrap."""

import argparse
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from .io import read, save


def normalize_number(s):
    s = s.replace(",", "").strip().strip("$").strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def math_correct(text, reference):
    target = normalize_number(reference.split("####")[-1])
    hits = re.findall(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", text)
    # Strict requested format, with a separately recorded fallback for inspection.
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    pred = normalize_number(hits[-1]) if hits else None
    fallback = normalize_number(numbers[-1]) if numbers else None
    return {
        "correct": pred is not None and pred == target,
        "predicted": str(pred),
        "target": str(target),
        "format_valid": bool(hits),
        "fallback_correct": fallback is not None and fallback == target,
    }


def isolate_code(code, test, entry):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", code, re.S)
    clean = blocks[0] if blocks else code
    # A function body without a definition is unsupported by this chat protocol.
    if not re.search(r"\bdef\s+" + re.escape(entry) + r"\s*\(", clean):
        return {"passed": False, "status": "missing_function_definition"}
    source = (
        "from typing import *\nimport math\n" + clean + "\n" + test + "\ncheck(" + entry + ")\n"
    )
    bwrap = os.environ.get("INFERBENCH_BWRAP") or shutil.which("bwrap")
    if os.name != "posix" or not bwrap or not Path(bwrap).is_file():
        return {"passed": False, "status": "sandbox_unavailable"}
    import resource

    with tempfile.TemporaryDirectory(prefix="infra_eval_") as td:
        p = Path(td) / "sample.py"
        p.write_text(source, encoding="utf-8")
        cmd = [
            bwrap,
            "--unshare-all",
            "--die-with-parent",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            str(p),
            "/sample.py",
            "--chdir",
            "/tmp",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin:/bin",
            "--",
            "/usr/bin/python3",
            "-I",
            "-c",
            'import resource,runpy;resource.setrlimit(resource.RLIMIT_NPROC,(64,64));runpy.run_path("/sample.py",run_name="__main__")',
        ]

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6, preexec_fn=limits)
            return {
                "passed": r.returncode == 0,
                "status": "passed" if r.returncode == 0 else "failed",
                "returncode": r.returncode,
                "stderr": r.stderr[-1600:],
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout"}


def evaluate(case_path, manifest_path, output):
    refs = {r["id"]: r.get("reference") for r in read(manifest_path)["rows"]}
    rows = []
    for repeat, rd in enumerate(read(case_path)["rounds"]):
        for row in rd["requests"]:
            reference = refs[row["id"]]
            if reference is None:
                raise ValueError("Manifest is missing task references")
            if row["task"] == "math":
                result = math_correct(row["text"], reference)
            elif row["task"] == "code":
                result = isolate_code(row["text"], reference["test"], reference["entry_point"])
            else:
                continue
            rows.append({"repeat": repeat, "id": row["id"], "task": row["task"], **result})
    save(
        output,
        {
            "rows": rows,
            "protocol": "Subset greedy exact-match / code pass fraction; not official full-benchmark scores. Missing sandbox is unavailable evaluation, not an incorrect model answer.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.case, args.manifest, args.output)
