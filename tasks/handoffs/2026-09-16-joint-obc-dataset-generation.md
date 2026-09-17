# Handoff: prepare a joint-estimation dataset for the gantry OBC experiment

**From**: session of 2026-09-16 | **Branch**: `Augmentation` | **Effort suggested**: `high`

## 1. Task

Assess the existing MATLAB gantry trajectory generator and propose the smallest defensible
configuration change for a dataset that supports joint estimation of the ten reduced physical
parameter combinations while retaining excitation of the hidden absorber used for augmentation.
The first phase is read-only. Present one concrete generation configuration to the user and wait
for explicit approval. Only after that approval may you edit MATLAB configuration and generate
data.

## 2. Out of scope

* Do not generate, simulate, overwrite, resume, or create any dataset before explicit user approval
  in the new session.
* Do not treat this handoff or the user's earlier request for a prompt as generation approval.
* Do not change Python training, OBC, encoder, or model-augmentation code.
* Do not redesign the 22 trajectory records unless evidence shows the current record family cannot
  excite one of the ten reduced parameter directions.
* Do not create a broad experiment campaign or compare every historical dataset.
* Do not modify `kamtin-fp-model/`.
* Do not overwrite an existing dataset folder.

## 3. Where things stand

The current branch is `Augmentation` at starting commit
`a52dd5584065f87bdeceef4b6f4a3ccdf05fb6cd`.

The working tree is heavily dirty. In particular,
`Matlab-scripts/Augmentation/data/generate_trajectory_data.m` and
`Matlab-scripts/Augmentation/data/gtd_config.m` contain uncommitted user work. Preserve it. Do not
restore or clean either file.

The active generated dataset is
`data/gantry/matlab/trajectory/augmentation_ma50_b140-230_a6_z03`. It contains 22 records and is
used by `scripts/gantry/gantry_interconnect_dynamic.py`. Its 14 training records are selected in
`scripts/gantry/gantry_dynamic/data.py`.

No MATLAB process or generation run is part of this handoff.

## 4. Established and verified

1. The MATLAB generator has no `joint_estimation` flag. Joint estimation is a Python training
   choice, not a data-generation mode.
2. MATLAB uses `TRACK` as its excitation selector in
   `Matlab-scripts/Augmentation/data/gtd_config.m`:
   * `joint` selects 1 to 200 Hz;
   * `joint_lowf` selects the record fundamental, approximately 0.0833 Hz, to 200 Hz;
   * `augmentation` selects 130 to 180 Hz before run-specific overrides.
3. The current driver selects `TRACK='augmentation'`, then overrides the band to 140 to 230 Hz,
   the excitation amplitude by a factor of 6, and absorber damping to 0.03.
4. The current 140 to 230 Hz dataset was designed to contain the absorber anti-resonance near
   150 Hz and the coupled pole near 212 Hz. D-188 records this choice.
5. `joint_lowf` was introduced because the baseline contains slow damping-related corners near
   0.10 and 0.16 Hz. A 1 Hz lower limit does not represent them well.
6. Extending a flat multisine band changes the number of excited lines and therefore the energy per
   line at fixed RMS. A wider band is not a free change.
7. The amplitude factor 6 headroom result was established for the 140 to 230 Hz configuration. It
   cannot be assumed unchanged for a much wider band without recalculating the limit pre-check.
8. `gtd_enforce_limits.m` performs the linear closed-loop hardware-limit pre-check and proportional
   scale-down. The current design aims to avoid record-dependent scaling so the dataset remains
   homogeneous.
9. The generator already refuses to write into a nonempty output directory when `RESUME=false`.

## 5. Assumed but not verified

1. The best single dataset is likely based on `TRACK='joint_lowf'` with the upper frequency
   extended far enough to contain the 212 Hz absorber pole. The exact upper limit must be justified.
2. The existing 22 record definitions probably provide enough Y-position and axis coverage for the
   ten reduced combinations. This must be assessed from their definitions and the planned OBC
   dataset-qualification diagnostics, not assumed from their names.
3. Keeping `USE_MSD=true`, `MA_FRAC=0.50`, and absorber damping 0.03 is probably necessary for a
   direct comparison with the current augmentation dataset.
4. Amplitude factor 6 may no longer be the largest homogeneous value after broadening the band.
5. One broadband dataset may be adequate for both physical parameter estimation and augmentation.
   If spectral dilution makes this implausible, report that conflict rather than silently creating
   two datasets.

## 6. Tried and failed

* The older `augmentation` default band of 130 to 180 Hz targeted the standalone absorber frequency
  and missed the actual coupled pole for a 50 percent absorber mass split.
* A 165 to 260 Hz augmentation band contained the pole but omitted the 150 Hz anti-resonance.
* The current 140 to 230 Hz band addresses the augmentation dynamics but was not designed to expose
  the low-frequency baseline damping directions needed for joint physical estimation.
* Calling the MATLAB `TRACK` selector a `joint_estimation` flag conflates experiment excitation with
  the estimator used later in Python.

## 7. Achieved

The existing generator already provides:

* deterministic track-specific seeds;
* 14 training, 4 validation and 4 test records;
* standstill, Y-sweep, mixed-axis, APRBS and Lissajous trajectories;
* closed-loop force injection;
* plant input, output, physical-state and hidden-absorber logging;
* hardware-limit pre-checking;
* a fresh-folder guard and resumable generation;
* per-record diagnostic figures.

The likely change is therefore configuration-level rather than a new generation pipeline.

## 8. The open question

Which single multisine band and amplitude best preserve low-frequency information for the reduced
physical parameters while retaining useful excitation at the 150 Hz anti-resonance and 212 Hz
coupled absorber pole?

The leading candidate is `joint_lowf` with its upper limit extended beyond 212 Hz. Do not lock in
0.0833 to 230 Hz merely because it combines the two existing endpoints. Assess line density,
per-line amplitude, frequency coverage, record duration and hardware headroom first.

## 9. Next action

Perform a read-only design audit and return exactly one recommended configuration. Inspect the
current MATLAB generator, record definitions, limit pre-check, D-188, and the reduced parameter
combinations. The recommendation must state:

* `TRACK`;
* `USE_MSD`;
* `MA_FRAC`;
* lower and upper multisine frequencies;
* whether record duration must change;
* absorber damping;
* excitation amplitude strategy;
* `SELECT`;
* `RESUME`;
* a new, collision-free `OUT_DIR_NAME` encoding the important settings;
* which existing MATLAB lines would change;
* expected runtime and disk use;
* why the proposed spectrum excites both the ten reduced baseline directions and the absorber;
* which properties remain uncertain until the non-writing limit calculation or first approved
  probe is performed.

Use analytical calculations and existing stored results where possible. Do not call
`generate_trajectory_data.m`, Simulink, or any command that writes trajectory files in this phase.

End phase one with a direct approval question containing the complete proposed configuration. Make
clear that approval authorizes the stated edits and the stated generation sequence. An acceptable
form is:

> I recommend generating `<folder>` with `<track>`, `<band>`, `<amplitude>`, and `<damping>`. This
> will first run `<probe or pre-check>` and will generate all 22 records only if its stated limit
> criterion passes. Do you approve this exact generation plan?

Do not proceed until the user explicitly approves it in that session.

After approval:

1. Log the accepted dataset design in `docs/decisions.md` before changing the generator.
2. Apply only the accepted MATLAB configuration changes. Preserve the previous settings in version
   history and use a new output directory.
3. Run the cheapest non-writing limit and spectrum check available. If it contradicts the approved
   configuration, stop and return the measured result instead of silently changing settings.
4. If the approved plan includes a probe record, generate only that probe and report its force,
   position, velocity, spectrum and absorber-response checks. Wait for further approval before a
   full 22-record batch unless the user's approval explicitly included both the probe and automatic
   continuation after its numerical gate passes.
5. Generate the full dataset only under the approved continuation rule.
6. Validate record count, lengths, finite values, limits, spectral coverage, output folder and
   metadata. Report any record-dependent scaling.

## 10. Acceptance criterion

Phase one is complete when the user receives one concrete, technically justified configuration and
an explicit generation approval question, with no files changed and no MATLAB generation started.

If the user later approves generation, the complete task is finished when:

* the accepted configuration is logged and represented in MATLAB;
* the fresh output folder contains the approved records and no older folder was modified;
* all generated signals are finite and have the expected dimensions and sample rate;
* every record satisfies the declared position, velocity, force-peak and force-RMS limits;
* any automatic scaling is reported by record and does not violate the approved homogeneity rule;
* the realized excitation covers the accepted frequency band and the absorber features;
* the final dataset path and exact settings are reported for the later Python OBC session.

## 11. Read these first

1. `Matlab-scripts/Augmentation/data/generate_trajectory_data.m` for the active run configuration
   and overwrite guard.
2. `Matlab-scripts/Augmentation/data/gtd_config.m` for track meanings, dynamics, rates and limits.
3. `Matlab-scripts/Augmentation/data/gtd_build_records.m` for the 22-record excitation design.
4. `docs/trajectory-generation-spec-draft.md` for the intended data-generation contract.
5. D-188 in `docs/decisions.md` for the purpose and limits of the current 140 to 230 Hz dataset.

## 12. Do not

* Do not interpret this prompt as approval to generate.
* Do not run MATLAB or Simulink before presenting the recommendation unless the command is proven
  read-only and cannot create caches, figures, logs, directories or trajectory files.
* Do not edit MATLAB files during phase one.
* Do not use an existing output folder for the new dataset.
* Do not set `RESUME=true` for a fresh dataset.
* Do not assume amplitude factor 6 remains safe after changing the band.
* Do not broaden the band without discussing energy dilution across frequency lines.
* Do not change the physical truth model merely to make joint estimation easier.
* Do not alter train, validation or test membership without evidence and explicit user approval.
* Do not launch Python training after generation.

## 13. Operational

Use the `GraduationProject` environment where Python inspection is needed and the installed MATLAB
version used by this repository for MATLAB execution. Follow the live-output convention for any
approved run lasting more than a few seconds.

The full batch can take hours and write hundreds of megabytes. Before an approved launch, report
the exact command, output directory, expected record count and log path. Stream progress to a log
the user can inspect.

## 14. Delegation

No subagent is needed. The relevant generator and design documents are targeted and bounded.

In the first response, lead with the distinction between Python joint estimation and MATLAB
`TRACK`, then give the single recommended configuration and its evidence. End with the approval
question. Do not end with a menu of bands.
