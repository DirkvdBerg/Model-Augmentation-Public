# Reference: where the verification material is

This folder holds the data generator and its documentation. Everything that verified it, logged its runs or launched it unattended was moved on 2026-09-27 to `scripts/gantry/thesis-data-verification/` (user decision, GENERATION-REPORT U8). Nothing there is needed to generate data.

## Contents of `scripts/gantry/thesis-data-verification/`
| Path | What it is |
|-|-|
| `checks/` | One MATLAB script per check C1 to C16 (criteria stated in each header before its first run), their result files, the noise calibration figure `fig_noise_calibration.png` and its Python script |
| `logs/` | Every MATLAB run: output (`<run>/attempt*.log`), watchdog record (`resources.csv`, `watchdog_summary.txt`); `generation_log.csv` (one line per generated record; the generator keeps writing here, `cfg.log_dir`); `generation_failures.csv`; `superseded/` for runs with settings later changed |
| `tools/` | `watchdog.ps1` (emergency kill at 1.0 GB available RAM), `run_matlab.sh` (one MATLAB job at a time under the watchdog, retry after an emergency kill, ends a MATLAB that hangs after finishing), `pipeline_pilot.sh` and `pipeline_campaign.sh` (the unattended sequences used on 2026-09-26) |
| `REVIEW.md` | The independent review of the generator and the pilot, and how each finding was handled |
| `vendor/original/` | Untouched copies of every source file the generator was copied from (provenance, diffs; `VENDORED.md` lists them with their hashes) |
| `vendor/` | `gantrySystem.m`, `gantrySystemCoriolisCentripetal.m`: needed only to load the original four-loop model in check C1 |
| `noise-twins/` | The 4 noise-only twins (E1, I1, I4, E5 `_nt1`), a second noise draw of those records; not part of the data set since the floor is defined as noisy minus noise-free (U7); kept as a cross-check |

## What the checks proved (details in `IMPLEMENTATION.md` section 5 and `GENERATION-REPORT.md` section 2)
- C1: the generator's model equals the original model bitwise with the noise off
- C2: the noise reproduces the measured error below 300 Hz within 1.4 % at every Y position used
- C3, C12, C14: every noise-free file equals its noisy twin outside the noise path; the plant received exactly the saved force plus the noise; every saved field is double and complete
- C4, C9, C10: the multisine has 383 lines on 106 to 297 Hz at 240 / 87 / 180; every reference is within the limits and built by the same rules as the design's excitation evidence
- C5, C6: every record passed the limit check on the simulated truth; K1 and K2-g+ are stable over Y = ±0.39
- C7, C8: seeds distinct; the absorber and friction parameters in every file equal the design values
- C11, C13: the noise calibration's linearisation and the noise draws are exact
- C15: MATLAB without its JVM gives the same three simulation checksums as with it (not a bitwise comparison of every array)
- C16: the current code regenerates saved records bitwise (TF-1_r1, TR-P3_r1, E5_r1, E6_r1), so the data do not depend on the code edits made after they were generated

## Rerunning a check
From the verification folder, one MATLAB job at a time: `tools/run_matlab.sh <name> "addpath('checks'); check_references"` (the launcher puts `Thesis-writeup/Code/Data` on the MATLAB path), or in MATLAB: `addpath('<repo>/Thesis-writeup/Code/Data'); addpath('checks'); check_references`
