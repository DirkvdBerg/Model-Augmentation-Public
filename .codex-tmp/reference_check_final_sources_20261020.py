from pathlib import Path
import requests,json,concurrent.futures
import pymupdf
out=Path('.codex-tmp/reference-check-20261020')
urls={
'official-ovchinnikov-final':'https://www.sciencedirect.com/science/article/pii/S0167691121001602/pdfft?isDTMRedir=true&download=true',
'official-garcia-final':'https://www.sciencedirect.com/science/article/pii/S0967066112002328/pdfft?isDTMRedir=true&download=true',
'official-ovchinnikov-full':'https://api.elsevier.com/content/article/pii/S0167691121001602?view=FULL&httpAccept=text/xml',
'official-willard-final':'https://dl.acm.org/doi/pdf/10.1145/3514228',
'official-willard-export':'https://dl.acm.org/action/exportCiteProcCitation?dois=10.1145%2F3514228&targetFile=custom-bibtex&format=bibTex',
'official-drenth-crossref-search':'https://api.crossref.org/works?query.title=Efficient%20Learning%20of%20Affine%20and%20Rational%20Dependency%20LPV&rows=3',
'official-encoder-crossref-search':'https://api.crossref.org/works?query.title=Encoder%20initialisation%20methods%20in%20the%20model%20augmentation%20setting&rows=3',
}
def inspect(item):
 name,url=item
 try:
  r=requests.get(url,timeout=25);r.raise_for_status()
  if r.content.startswith(b'%PDF'):
   d=pymupdf.open(stream=r.content,filetype='pdf');t='\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+p.get_text(sort=True) for i,p in enumerate(d))
  else:t=r.text
  (out/(name+'.txt')).write_text(t,encoding='utf-8')
  if 'crossref-search' in name:
   return name+' '+json.dumps([{k:m.get(k) for k in ['title','DOI','published']} for m in r.json()['message']['items']],ensure_ascii=False)
  return f'{name}: {r.status_code} {len(r.content)} bytes; {t[:300]}'
 except Exception as e:return name+': '+str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for result in ex.map(inspect,urls.items()):print(result)
