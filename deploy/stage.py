"""Copy exactly what the Lambda function needs into .aws-sam/stage/, for `sam build` (spec step 8).

From the repo root:  python -m deploy.stage && sam build && sam deploy

Why a staging folder: template.yaml's CodeUri is the folder SAM packages. Pointing it at the repo root
would also package the corpus, eval, tests and whatever else is lying in the folder (your .venv,
scratch/). The function needs only:
    app/*.py                        the code (search, prompt, answer flow, tool, citation guard, handler)
    index/chunks.json, embeddings.npy, meta.json    the search index (about 9 MB)
    data/us_federal_holidays.json   the tool's business-day table
    requirements.txt                SAM installs these into the package (boto3, numpy, pydantic)
The folder keeps the repo's layout, because app/retrieve.py and app/tools.py find index/ and data/
next to app/. .aws-sam/ is already in .gitignore, so nothing here is committed.

The copied requirements must not ask for botocore[crt] or awscrt: the laptop needs CRT to read
`aws login` sessions, but the function gets its credentials from its role (spec).
"""
import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".aws-sam" / "stage"
FILES = ["index/chunks.json", "index/embeddings.npy", "index/meta.json", "data/us_federal_holidays.json",
         "requirements.txt"]
CRT = re.compile(r"awscrt|\[\s*crt\s*\]", re.IGNORECASE)


def app_files(root=ROOT):
    return sorted(p.relative_to(root).as_posix() for p in (root / "app").glob("*.py"))


def check_requirements(text):
    """Raise ValueError if a requirement line (comments ignored) asks for the CRT."""
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line and CRT.search(line):
            raise ValueError(f"requirements.txt asks for the AWS CRT ({line!r}); the function must not ship it")


def stage(root=ROOT, dest=None):
    """Recreate the staging folder. Returns [(relative path, bytes, sha256 first 8)] in copy order."""
    dest = Path(dest) if dest else root / ".aws-sam" / "stage"
    check_requirements((root / "requirements.txt").read_text(encoding="utf-8"))
    if not (root / "app" / "handler.py").exists():
        raise FileNotFoundError("app/handler.py is missing")
    if dest.exists():
        shutil.rmtree(dest)
    copied = []
    for rel in app_files(root) + FILES:
        src, out = root / rel, dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        data = out.read_bytes()
        copied.append((rel, len(data), hashlib.sha256(data).hexdigest()[:8]))
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description="Copy the function's files into .aws-sam/stage/ for sam build.")
    ap.parse_args(argv)
    try:
        copied = stage()
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"Not staged: {e}")
    for rel, size, digest in copied:
        print(f"  {rel:40} {size:>10,} bytes  {digest}")
    print(f"Staged {len(copied)} files, {sum(s for _, s, _ in copied) / 1e6:.1f} MB, in {STAGE.relative_to(ROOT)}/")
    print("Next: sam build")


if __name__ == "__main__":
    main()
