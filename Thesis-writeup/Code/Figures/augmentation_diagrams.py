"""Build the thesis-owned common-reference figure for Section III.

The author edited Writing/figures/tex/augmentation_structure.tex directly:
never overwrite it with the older Code/Figures source. The horizon diagram
and older condensed residual-loop diagram are not included in the section.
Run from any directory: python Thesis-writeup/Code/Figures/augmentation_diagrams.py
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = Path(__file__).resolve().parent
DESTINATION = SOURCE.parents[1] / "Writing" / "figures"
FIGURE = "closed_loop_same_reference"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    destination = DESTINATION / f"{FIGURE}.pdf"
    with tempfile.TemporaryDirectory(prefix="thesis-closed-loop-") as scratch:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={scratch}", f"{FIGURE}.tex"],
            cwd=SOURCE, check=True,
        )
        shutil.copyfile(Path(scratch) / f"{FIGURE}.pdf", destination)
    print(f"Published {destination}; author's architecture left unchanged")


if __name__ == "__main__":
    main()
