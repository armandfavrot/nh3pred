"""Execute installed-package examples and render self-contained vignettes.

Requires the docs extra, Quarto and pdflatex. Never reads research datasets.
All intermediate files and the Jupyter kernel are isolated in a temporary folder.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    docs = Path(__file__).resolve().parent
    destination = docs / "source" / "_static" / "vignettes"
    names = [
        "getting_started",
        "input_reference",
        "scenario_comparison",
        "model_description",
    ]
    titles = [
        "Getting started",
        "Input reference",
        "Scenario comparison",
        "Model and interpretation",
    ]
    with tempfile.TemporaryDirectory(prefix="nh3pred-vignettes-") as temporary:
        root = Path(temporary)
        work = root / "sources"
        shutil.copytree(docs / "vignettes", work)
        kernel = root / "jupyter" / "kernels" / "python3"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "nh3pred documentation",
                    "language": "python",
                }
            )
        )
        env = os.environ.copy()
        env.update(
            QUARTO_PYTHON=sys.executable,
            JUPYTER_PATH=str(root / "jupyter"),
            JUPYTER_RUNTIME_DIR=str(root / "runtime"),
            MPLBACKEND="Agg",
            MPLCONFIGDIR=str(root / "matplotlib"),
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            XDG_CACHE_HOME=str(root / "cache"),
            XDG_DATA_HOME=str(root / "data"),
        )
        for name in names:
            subprocess.run(
                ["quarto", "render", name + ".qmd", "--to", "html"],
                cwd=work,
                env=env,
                check=True,
            )
        combined = """---
title: "nh3pred 0.2.0: vignettes"
format:
  pdf:
    pdf-engine: pdflatex
    latex-auto-install: false
jupyter: python3
execute:
  warning: false
---
"""
        for name, title in zip(names, titles):
            body = (work / (name + ".qmd")).read_text().split("---", 2)[2]
            combined += "\n\\newpage\n\n# " + title + "\n" + body
        (work / "ammonia_emissions_vignettes.qmd").write_text(combined)
        subprocess.run(
            ["quarto", "render", "ammonia_emissions_vignettes.qmd", "--to", "pdf"],
            cwd=work,
            env=env,
            check=True,
        )
        destination.mkdir(parents=True, exist_ok=True)
        for filename in [
            *(name + ".html" for name in names),
            "ammonia_emissions_vignettes.pdf",
            "styles.css",
        ]:
            shutil.copy2(work / filename, destination / filename)
    print(f"Rendered vignettes: {destination}")


if __name__ == "__main__":
    main()
