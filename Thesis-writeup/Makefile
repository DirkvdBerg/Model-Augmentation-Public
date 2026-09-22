# Build for the 12-page IEEE paper.
# This file is the ONLY place that knows about both sides of the boundary:
# Code/Figures produces into Writing/figures, LaTeX consumes from there.
# Writing/ itself stays self-contained and portable (no ".." in any .tex).

PY      := conda run --no-capture-output -n GraduationProject python
FIGDIR  := Writing/figures
SRCDIR  := Code/Figures

# One stem per figure. Add a stem here when you add Code/Figures/<stem>.py.
STEMS   :=
FIGS    := $(addprefix $(FIGDIR)/,$(addsuffix .pdf,$(STEMS)))

.PHONY: all paper figures clean distclean

all: paper

# Pattern rule: Code/Figures/<stem>.py + Results/<stem>.csv -> figures/<stem>.pdf
$(FIGDIR)/%.pdf: $(SRCDIR)/%.py Results/%.csv
	$(PY) $<

figures: $(FIGS)

paper: figures
	cd Writing && latexmk -pdf -synctex=1 -outdir=build main.tex

clean:
	cd Writing && latexmk -c -outdir=build

distclean:
	cd Writing && latexmk -C -outdir=build
