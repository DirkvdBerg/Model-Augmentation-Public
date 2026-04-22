import re, collections, sys

path = "simulations/param_recovery/profile_trace.json"

totals = collections.defaultdict(float)
counts = collections.defaultdict(int)

event_lines = []
in_event = False
n_events = 0

print("Parsing trace...", flush=True)

with open(path, "r", encoding="utf-8") as f:
    for lineno, line in enumerate(f):
        s = line.rstrip("\r\n")
        stripped = s.strip()

        # Event start: a line that is just "  {" (two spaces + brace)
        if stripped == "{" and s.startswith("  "):
            in_event = True
            event_lines = []
            continue

        if not in_event:
            continue

        # Event end
        if stripped in ("}", "},"):
            blob = " ".join(event_lines)
            name_m = re.search(r'"name":\s*"([^"]+)"', blob)
            dur_m  = re.search(r'"dur":\s*([\d.]+)', blob)
            cat_m  = re.search(r'"cat":\s*"([^"]+)"', blob)
            if name_m and dur_m:
                cat = cat_m.group(1) if cat_m else "?"
                key = f"[{cat}] {name_m.group(1)}"
                totals[key] += float(dur_m.group(1))
                counts[key] += 1
                n_events += 1
            in_event = False
            event_lines = []
        else:
            event_lines.append(stripped)

        if lineno % 1_000_000 == 0 and lineno > 0:
            print(f"  ...{lineno//1_000_000}M lines processed, {n_events} events so far", flush=True)

print(f"\nDone. {n_events} timed events, {len(totals)} unique op types.")
grand = sum(totals.values())
print(f"Grand total duration captured: {grand/1e6:.1f} s\n")

print("=== TOP 50 OPS BY TOTAL DURATION ===")
for key, dur in sorted(totals.items(), key=lambda x: -x[1])[:50]:
    pct = 100 * dur / grand
    avg = dur / counts[key]
    print(f"{dur/1e6:10.3f} s  {pct:5.1f}%  avg {avg/1e3:8.1f} ms  ({counts[key]:>9,} calls)  {key}")
