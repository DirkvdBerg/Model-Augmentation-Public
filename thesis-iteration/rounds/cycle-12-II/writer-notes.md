# Writer notes, cycle 12 (Section II), 2026-10-09

1. Contradiction: README row II-A says "no entry states why the baseline omits" Coulomb friction, but D-204 (listed in that row) has an explicit "Why friction in the truth only"; the skill says a reason found in the row is stated, the README says todo. Its reason needs the Section V truth (Known before new), so I wrote a todo naming D-204 vs D-022.
2. Unclear: "Section II names M_LPV and M_LTI together in a closing part": paragraph or subsection, and which equations make them "complete" (CT EOM, or the RK4 model of Section III)? I made subsection II-D with the CT EOM plus output map, nominal theta.
3. Silent: no MUST ESTABLISH block for II-A and II-D; I inferred PROPOSED blocks (skill 2a) but nothing says what minimum such a block must hold.
4. Contradiction: agreed II-B block cites "Drenth Eq. (6) form"; README row says cite drenth2025thesis instead. Corrected per skill 2a, named in header. The block also writes N(Y)/d(Y), where d already names the payload offset; one-meaning rule forced adj/det wording.
5. Silent: the "at most one sentence after a where clause" rule versus a where clause that must also carry the nominal-value pointer for theta; I folded it into the where clause in parentheses.
6. Unclear: L_b fixed: the reason (P and q independent of estimated parameters) is visible in the display but no decision states it; the rules on self-derived reasons (todo) vs display-shown losses (state it) leave this ambiguous. I stated it.
7. Silent: an existing header contains "--" (G--Delta); the no-em-dash rule vs "keep the header" conflicts. I changed it to "G-Delta".
8. Unclear: header Caution "resolve the supervisor comment on dual-gantry terminology" has no findable source; covered as candidate todo (Garcia's "dual-drive gantry").
9. Silent: II-C block says "no new symbols", but the Jacobian argument needs entry names (M_{0,33} etc.); unclear whether indexed entries count as new symbols.
10. Length test is impractical: "compile a draftfalse copy, figures counted" gives only page ship-out boundaries; I measured about 2 pages vs 1.35 budget (about 1.5x), so the second deletion pass trigger was borderline.
11. Known before new vs README ownership: M_LPV/M_LTI purpose must use THESIS-RESULTS terms ("demonstration because the truth contains LPV structure"), but the truth is a Section V fact; I kept only "demonstration of the size of the position-dependent error".
12. Notation clash m_Delta / xi_Delta vs scheduling block Delta: the rules require a todo in this section but give no candidate convention; I proposed m_d.
13. Tooling trap (not a rule): heredocs in this Bash tool collapse "\\" to "\", which silently broke LaTeX matrix rows; writers should edit .tex with the Edit tool.
