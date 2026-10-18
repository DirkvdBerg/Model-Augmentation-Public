from pathlib import Path
import json,urllib.request,urllib.parse,time
OUT=Path(__file__).resolve().parent
entries=json.loads((OUT/'bibliography.json').read_text(encoding='utf-8'))
dest=OUT/'external-sources';dest.mkdir(exist_ok=True)
urls={
 'ovchinnikov2021computing-v3.pdf':'https://arxiv.org/pdf/2004.07774v3',
 'gyorok2025l4dc-published.pdf':'https://raw.githubusercontent.com/mlresearch/v283/main/assets/gyorok25a/gyorok25a.pdf',
 'gautier1990minimum.pdf':'https://hal.science/hal-05223021/document',
}
log=[]
for name,url in urls.items():
 try:
  if not (dest/name).exists():urllib.request.urlretrieve(url,dest/name)
  log.append({'artifact':name,'url':url,'status':'downloaded','bytes':(dest/name).stat().st_size})
 except Exception as e:log.append({'artifact':name,'url':url,'error':str(e)})
for e in entries:
 if e['file']!='refs.bib' or not e['cited_main']:continue
 doi=e['fields'].get('doi')
 if not doi or 'arXiv' in doi:continue
 url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
 name=e['key']+'-crossref.json'
 try:
  with urllib.request.urlopen(url,timeout=40) as h:data=json.load(h)
  (dest/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
  log.append({'artifact':name,'url':url,'status':'downloaded'})
 except Exception as ex:log.append({'artifact':name,'url':url,'error':str(ex)})
(OUT/'retrieval-log.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
print(json.dumps(log,indent=2))
