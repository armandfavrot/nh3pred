"""Execute installed-package examples and render self-contained HTML vignettes.

Requires the docs extra and Quarto. Never reads research datasets.
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
        destination.mkdir(parents=True, exist_ok=True)
        for filename in [
            *(name + ".html" for name in names),
            "styles.css",
        ]:
            shutil.copy2(work / filename, destination / filename)
    print(f"Rendered vignettes: {destination}")


if __name__ == "__main__":
    main()
