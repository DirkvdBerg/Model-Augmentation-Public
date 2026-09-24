"""Session resource summary over every outputs/<run>/resources.csv (for REPORT.md)."""
import csv
import glob
import os

TR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(os.path.join(TR, 'outputs', '*', 'resources.csv'))):
    run = os.path.basename(os.path.dirname(f))
    with open(f) as fh:
        r = list(csv.DictReader(fh))
    if not r:
        continue
    rows.append(dict(run=run, peak_ws=max(float(x['tree_ws_mb']) for x in r),
                     min_ram=min(float(x['avail_ram_gb']) for x in r),
                     min_disk=min(float(x['disk_free_gb']) for x in r),
                     kills=sum('kill' in x['flag'] for x in r),
                     alerts=sum('alert' in x['flag'] for x in r)))
nk = [x for x in rows if x['kills'] and x['run'] != 'g0_kill']
top = max(rows, key=lambda x: x['peak_ws'])
print(f'runs {len(rows)}; peak tree working set {top["peak_ws"]:.0f} MB ({top["run"]}); '
      f'min available RAM {min(x["min_ram"] for x in rows):.2f} GB; min C: free '
      f'{min(x["min_disk"] for x in rows):.2f} GB; watchdog kills outside the G0 kill test: {len(nk)}; '
      f'RAM-alert samples {sum(x["alerts"] for x in rows)}')
for x in rows:
    print(f'  {x["run"]:28s} peak {x["peak_ws"]:6.0f} MB  min RAM {x["min_ram"]:.2f} GB  '
          f'min C: {x["min_disk"]:.2f} GB  kills {x["kills"]}')
