"""Compile the TikZ slide figures in tikz/ to PNG in figures/.

    python build_tikz.py                     # all
    python build_tikz.py projection-sketch   # one, by output name

Compiles in a temporary directory (no LaTeX build output in the repo) with pdflatex, then
rasterises with pdftocairo at 600 dpi on a white background. The gantry schematics \\input the
thesis figure, which is copied into the temporary directory: read, never edited.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

from style import FIGDIR, HERE, THESIS_GANTRY_TEX

TIKZ = HERE / 'tikz'
DPI = 600
# output name -> (source, TeX definitions placed before it)
JOBS = {
    'gantry-baseline': ('gantry_baseline.tex', ''),
    'gantry-absorber': ('gantry_absorber.tex', ''),
    'equation': ('equation.tex', ''),
    'augmentation-step1': ('augmentation_steps.tex', r'\def\augstep{1}'),
    'augmentation-step2': ('augmentation_steps.tex', r'\def\augstep{2}'),
    'augmentation-step3': ('augmentation_steps.tex', r'\def\augstep{3}'),
    'augmentation-step4': ('augmentation_steps.tex', r'\def\augstep{4}'),
    'orthogonality': ('orthogonality.tex', ''),
    'orthogonality-formulas': ('orthogonality_formulas.tex', ''),
    'projection-sketch': ('projection_sketch.tex', ''),
    'validation-loop': ('validation_loop.tex', ''),
}


def build(names):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        shutil.copy(THESIS_GANTRY_TEX, tmp)
        for f in TIKZ.glob('*.tex'):
            shutil.copy(f, tmp)
        for name in names:
            src, defs = JOBS[name]
            job = 'job_' + name.replace('-', '_')
            (tmp / (job + '.tex')).write_text('%s\\input{%s}\n' % (defs, src[:-4]))
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                job + '.tex'],
                               cwd=tmp, capture_output=True, text=True, errors='replace')
            if r.returncode != 0:
                print(r.stdout[-3000:])
                raise SystemExit('pdflatex failed on %s' % src)
            pdf = tmp / (job + '.pdf')
            subprocess.run(['pdftocairo', '-png', '-r', str(DPI), '-singlefile', str(pdf),
                            str(FIGDIR / name)], check=True)
            print('[fig] figures/%s.png' % name)


if __name__ == '__main__':
    build(sys.argv[1:] or list(JOBS))
