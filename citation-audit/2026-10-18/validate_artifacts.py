from pathlib import Path
import json,hashlib
from html.parser import HTMLParser
O=Path(__file__).resolve().parent
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':self.links.extend(v for k,v in attrs if k=='href')
p=Links();p.feed((O/'report.html').read_text(encoding='utf-8'))
local=[u for u in p.links if not u.startswith(('http','#'))]
missing=[u for u in local if not (O/u.split('#')[0]).exists()]
rows=json.loads((O/'occurrence-assessments.json').read_text(encoding='utf-8'));bad=[]
for r in rows:
 for ref in r['evidence'].split(';'):
  if not ref:continue
  name,rng=ref.split(':');end=int(rng.split('-')[-1])
  if end>len((O/'source-text'/f'{name}.txt').read_text(encoding='utf-8').splitlines()):bad.append((r['id'],ref))
initial=json.loads((O/'initial-input-manifest.json').read_text(encoding='utf-8'));repo=O.parents[1]
drift=[x['path'] for x in initial if hashlib.sha256((repo/x['path']).read_bytes()).hexdigest()!=x['sha256']]
v={'report_local_links_checked':len(local),'missing_local_links':missing,'out_of_range_evidence':bad,'editor_occurrences_assessed':sum(x['main_visible'] for x in rows),'primary_pdf_hashes_unchanged':not any(x.startswith('literature/') for x in drift),'live_changes_since_initial_inventory':drift,'live_changes_after_snapshot':[p.relative_to(O/'snapshot/Writing').as_posix() for p in (O/'snapshot/Writing').rglob('*.tex') if p.read_bytes()!=(repo/'Thesis-writeup/Writing'/p.relative_to(O/'snapshot/Writing')).read_bytes()]}
(O/'validation.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(v,indent=2));assert not missing and not bad;assert v['primary_pdf_hashes_unchanged']
