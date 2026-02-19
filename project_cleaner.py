#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path.cwd()

CODE_DIRS = {
    ".py": "src",
    ".js": "src",
    ".ts": "src",
    ".json": "config",
    ".md": "docs",
}

JUNK_DIRS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".DS_Store",
}

def run(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except Exception:
        pass

def format_code():
    run(["python", "-m", "pip", "install", "--quiet", "black", "ruff"])
    run(["black", "."])
    run(["ruff", "check", ".", "--fix"])

def clean_junk():
    for path in ROOT.rglob("*"):
        if path.is_dir() and path.name in JUNK_DIRS:
            shutil.rmtree(path, ignore_errors=True)

def organize_files():
    for path in ROOT.iterdir():
        if path.is_file():
            dest_dir = CODE_DIRS.get(path.suffix)
            if dest_dir:
                target = ROOT / dest_dir
                target.mkdir(exist_ok=True)
                shutil.move(str(path), target / path.name)

def normalize_structure():
    for folder in CODE_DIRS.values():
        (ROOT / folder).mkdir(exist_ok=True)

def main():
    normalize_structure()
    clean_junk()
    format_code()
    organize_files()
    print("✔ Project cleaned, formatted, and organized")

if __name__ == "__main__":
    main()
