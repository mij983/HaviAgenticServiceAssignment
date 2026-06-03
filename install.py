"""
install.py
-----------
Creates a virtual environment at .venv/ and installs all dependencies.

Run with system Python (no activation needed):
    python3 install.py

What it does:
  1. Creates .venv/ using  python3 -m venv  (subprocess, not the venv module)
  2. Upgrades pip inside the venv
  3. Installs CPU-only PyTorch (~180MB) via the venv's pip
  4. Installs all remaining packages via the venv's pip

After it finishes, activate the venv:
  Mac/Linux : source .venv/bin/activate
  Windows   : .venv\\Scripts\\activate
"""

import subprocess
import sys
from pathlib import Path

VENV_DIR = Path(".venv")


# ── helpers ───────────────────────────────────────────────────────────────────

def venv_python() -> str:
    if sys.platform == "win32":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python3")


def venv_pip() -> str:
    if sys.platform == "win32":
        return str(VENV_DIR / "Scripts" / "pip.exe")
    return str(VENV_DIR / "bin" / "pip3")


def run(cmd: list, label: str):
    print(f"\n  Installing: {label}...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  [ERROR] Failed: {label}")
        print("  Try manually: " + " ".join(cmd))
        sys.exit(1)
    print(f"  [OK] {label}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  ARIA -- Installing Dependencies")
    print("  CPU-only install (no GPU required)")
    print("=" * 60)

    # Step 1 — create venv using subprocess so it never touches system pip
    print("\n  Step 1/4: Creating virtual environment at .venv/")
    if VENV_DIR.exists():
        print("  [SKIP] .venv/ already exists — reusing it.")
        print("  Delete .venv/ and re-run to do a clean install.")
    else:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print("  [OK] Virtual environment created.")

    # Step 2 — upgrade pip inside the venv
    print("\n  Step 2/4: Upgrading pip inside the venv")
    subprocess.run(
        [venv_python(), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        check=True,
    )
    print("  [OK] pip upgraded.")

    # Step 3 — CPU-only PyTorch (needs special index URL)
    print("\n  Step 3/4: Installing CPU-only PyTorch (~180MB)")
    print("  (avoids the 2GB GPU build)")
    run(
        [
            venv_pip(), "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet",
        ],
        "torch (CPU only)",
    )

    # Step 4 — everything else
    print("\n  Step 4/4: Installing remaining packages")

    packages = [
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "chromadb>=0.5.0",
        "ollama>=0.2.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-ollama>=0.2.0",
        "flask>=3.0.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.0",
        "pypdf>=4.0.0",
        "beautifulsoup4>=4.12",
    ]

    for pkg in packages:
        run([venv_pip(), "install", pkg, "--quiet"], pkg)

    # Done
    print()
    print("=" * 60)
    print("  All packages installed successfully.")
    print("=" * 60)
    print()
    if sys.platform == "win32":
        activate = r"  .venv\Scripts\activate"
    else:
        activate = "  source .venv/bin/activate"

    print("  NEXT — activate the venv, then use the project:")
    print()
    print(activate)
    print()
    print("  ollama pull gemma3:4b               # pull LLM model")
    print("  python build_knowledge_base.py      # build KB from CSV")
    print("  python app.py                       # start web UI")
    print()


if __name__ == "__main__":
    main()
