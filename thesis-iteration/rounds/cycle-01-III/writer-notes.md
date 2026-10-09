# Writer notes, cycle 01, Section III (rules evidence)

1. Contradiction: SKILL step 3 says "Do not compile"; the run task requires a compile. Followed the task.
2. Ownership conflict: README gives III the optimiser and checkpoint selection, but sources.md section 3 tags "Training loop, polish" and "Validation, checkpoint selection" as (V), and 05_setup.tex still holds the free-run validation bullet. Wrote both in III-C; 05 left untouched (flag).
3. Contradiction on W^a: sources.md section 1 says Xavier (D-239), its section 4 superseded table says "code uses Kaiming", and PS2-METHOD.md says Kaiming. Code (`_THESIS_FIXED`, `wa_encoder_init='xavier_uniform_gain1'`) wins; rules should mark PS2-METHOD Sec. 2.1 stale.
4. Silent: model-name notation. README's M_U clashes with the inertia matrix M(Y); used \mathcal M_U, \mathcal M_PS2 plus a todo.
5. Silent: how to handle a pre-existing MUST ESTABLISH block that misses parts (III-D, optimiser, selection) when nobody can approve. Appended a "PROPOSED ADDITIONS, not approved" block.
6. Unclear: README's PS2 sketch ("about two displays and six sentences") versus "make clear what is trained, fixed, against which cost". S1 target, trained parameters, head, switching and S2 needed about 20 sentences; is the sketch a target or a cap?
7. Unclear: the packing check (no sentence over 30 words outside a "where" clause) flags every lead-in plus display plus where clause; treated display sentences as exempt.
8. Silent: whether decision IDs (D-220) or code paths may appear in prose. Kept them only inside \todo.
9. Silent: whether the writer may add refs.bib entries for papers the source map marks "no key" (Hefny, Downey) or that 04 already cites (gyorok2026obc). Added four verified entries with % CHECK comments.
10. Conflict: header Caution "distinguish the truth-only hidden absorber from learned states" versus the skill's "argue from the general case" (absorber appears only in Experiment design). Wrote the general sentence (states not tied to a subsystem).
11. Conflict: Research-plan anchor "document the departure from earlier state routing" versus the reader test (no development history). Stated only the final routing.
12. Silent: where code defects found while verifying go (oracle polish meter `combo_err`, S2 rollback defect, D-142 vs closed_loop.py comment). Put each in a \todo at its paragraph.
13. Source map gap: row III-C does not list D-229 or `Reduced_Gantry_State_Block` although Section II moves the m_Delta training map into III-C; `sutanto2020encoding` has no references.md row and stays unverified.
