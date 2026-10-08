#!/bin/bash
# Submit one tier of the thesis campaign (Thesis-writeup/Documentation/TRAINING-DESIGN.md section 4)
# as a throttled job array on hawaii and oahu. Run from anywhere on the cluster:
#
#   bash scripts/gantry/thesis-results/submit.sh smoke         3 short launch tests on hawaii
#   bash scripts/gantry/thesis-results/submit.sh core [PAR]    the 13-run core set, ids 41 to 53 in priority order:
#                                                              R4/R6 seeds 1-3 (41-46), R2 0/4/8 states (47-49),
#                                                              noise-free pair (50-51), correct start (52-53)
#   THESIS_NITS=<n> bash .../submit.sh 4 [PAR]                 tier 4: window sweep 0.2 and 0.4 s (2 runs),
#                                                              n = pilot's plateau updates x 1.5 (TRAINING-DESIGN 1.1)
#   Tier "karnopp" (ids 1-28) ran on the superseded Karnopp data (array 86449); never resubmit it.
#   Extensions (seeds 4-5, more noise-free twins) get NEW ids; never reuse or renumber an id.
#
# PAR = at most this many runs of the tier at once (default 2), so the tier never claims every GPU.
# Logs: /home/dirk_van_den_berg/logs/augmentation/thesis-results/thesis-<tier>_<array>_<run id>.out
set -eo pipefail
cd "$(dirname "$0")/../../.."
TABLE=scripts/gantry/thesis-results/runs.tsv
SBATCH_FILE=scripts/gantry/thesis-results/run_thesis.sbatch
mkdir -p /home/dirk_van_den_berg/logs/augmentation/thesis-results

TIER="${1:?give a tier: smoke, 1, 2, 3 or 4}"
PAR="${2:-2}"
IDS=$(awk -F'\t' -v t="$TIER" 'NR>1 && $2==t {printf "%s%s", sep, $1; sep=","}' "$TABLE")
if [ -z "$IDS" ]; then echo "no runs for tier ${TIER} in ${TABLE}"; exit 2; fi
echo "tier ${TIER}: run ids ${IDS}"

case "$TIER" in
  smoke)
    # Short launch test (about 5 to 10 min): real data cut to 2 s, real GPU, eager, OBC, float64,
    # 4 updates, a 2-iteration polish and the end-of-run evaluation. Compilation is not in it:
    # watch the first real run's log for "training rollout COMPILED" and its first updates.
    sbatch -J thesis-smoke -p hawaii -t 00:30:00 --array="${IDS}" \
      --export=ALL,RUN_TABLE=${TABLE},THESIS_SMOKE=1 "$SBATCH_FILE" ;;
  4)
    : "${THESIS_NITS:?tier 4 needs THESIS_NITS = the pilot's plateau updates x 1.5 (TRAINING-DESIGN 1.1)}"
    sbatch -J thesis-t4 -t 48:00:00 --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE},THESIS_NITS=${THESIS_NITS} "$SBATCH_FILE" ;;
  *)
    sbatch -J "thesis-t${TIER}" --array="${IDS}%${PAR}" \
      --export=ALL,RUN_TABLE=${TABLE} "$SBATCH_FILE" ;;
esac
