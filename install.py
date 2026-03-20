"""
install.py
-----------
Run this instead of pip install -r requirements.txt

Installs CPU-only PyTorch first (small download ~180MB)
then installs the rest of the packages.

Usage:
    python install.py
"""

import subprocess
import sys


def run(cmd: list[str], label: str):
    print("")
    print("  Installing: " + label + "...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("")
        print("  [ERROR] Failed to install: " + label)
        print("  Try running manually:")
        print("  " + " ".join(cmd))
        sys.exit(1)
    print("  [OK] " + label)


def main():
    print("")
    print("=" * 60)
    print("  ARIA -- Installing Dependencies")
    print("  CPU-only install (no GPU required)")
    print("=" * 60)
    print("")
    print("  Step 1/2: Installing CPU-only PyTorch (~180MB)")
    print("  This avoids downloading the 2GB GPU version.")
    print("")

    # Step 1 — CPU-only torch (must come first)
    run(
        [
            sys.executable, "-m", "pip", "install",
            "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet",
        ],
        "torch (CPU only)",
    )

    print("")
    print("  Step 2/2: Installing remaining packages")
    print("")

    # Step 2 — everything else
    packages = [
        "sentence-transformers>=2.7.0",
        "faiss-cpu>=1.8.0",
        "chromadb>=0.5.0",
        "ollama>=0.2.0",
        "pandas>=2.0.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.0",
    ]

    for package in packages:
        run(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            package,
        )

    print("")
    print("=" * 60)
    print("  All packages installed successfully.")
    print("=" * 60)
    print("")
    print("  Next steps:")
    print("")
    print("  1. Install Ollama from https://ollama.com")
    print("     Then run:  ollama pull gemma:2b")
    print("")
    print("  2. Add your 1 year ServiceNow CSV to:")
    print("     data/training_tickets.csv")
    print("     Columns: Short Description, Description, Assignment Team")
    print("")
    print("  3. Build the knowledge base:")
    print("     python build_knowledge_base.py")
    print("")
    print("  4. Start predicting:")
    print("     python predict.py")
    print("")


if __name__ == "__main__":
    main()
