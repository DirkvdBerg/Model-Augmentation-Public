# Handoff: fix the typesetting of eq:obc_overlap
**Effort suggested**: medium (one display equation, LaTeX layout only)

## Task
In `Thesis-writeup/Writing/sections/04_obc.tex`, equation `\label{eq:obc_overlap}` (Section V-D), make the first line typeset cleanly while keeping exactly this notation:

ρ(Z) = max_{Q ∈ ran Φ ∖ {0}} ⟨Q, Z⟩ / (‖Q‖₂ ‖Z‖₂) = ‖ΠZ‖₂ / ‖Z‖₂

Problem: the subscript under `\max` is much wider than "max", so LaTeX leaves a large horizontal gap around it. The current attempt is `\max_{\mathclap{Q\in\ran\Phi\smallsetminus\{0\}}}`, with `\usepackage{mathtools}` added in `util/include.tex:6`. It has not been checked in a build yet.

## Out of scope
- The symbols (Q, Z, Π, `\smallsetminus`), the second line Π = Φ(ΦᵀΦ)⁻¹Φᵀ, and all prose around the equation.
- Any other equation or section.
- Change marks: there are none inside math.

## How to check
Dirk's editor rebuilds the thesis on save. Ask him to save, then render the page of `Thesis-writeup/Writing/build/main.pdf` that contains "Overlap with the parameter directions" (PyMuPDF, about 200 dpi) and look at the equation. Do not run a LaTeX build yourself unless Dirk asks for one.

## Acceptance criterion
The notation above appears unchanged, with no large gap between "ρ(Z) =" and the fraction, and the subscript touches neither the "=" nor the fraction. Show Dirk the rendered crop and the one-line LaTeX change, and apply only on his "yes".

## Delegation
None.
