# Build outside OneDrive (D-277): OneDrive locks files in a synced build folder
# while it uploads them, and pdflatex then aborts with "I can't write on file".
# Must match latex-workshop.latex.outDir in both .vscode/settings.json files; the
# editor recipe copies main.pdf, main.synctex.gz and main.log back to build/.
$out_dir = 'C:/Users/20203253/Documents/School/AI & Engineering systems/2025-2026/Thesis/build';
$aux_dir = 'C:/Users/20203253/Documents/School/AI & Engineering systems/2025-2026/Thesis/build';
$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode %O %S';
