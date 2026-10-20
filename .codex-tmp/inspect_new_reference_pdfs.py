from pathlib import Path
import hashlib,json,re
import pymupdf
folder=Path('literature/thesis-reference-new')
out=Path('.codex-tmp/reference-check-20261020/new-local')
out.mkdir(exist_ok=True)
records=[]
for p in sorted(folder.glob('*.pdf')):
    d=pymupdf.open(p)
    pages=[page.get_text(sort=True) for page in d]
    t='\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+page for i,page in enumerate(pages))
    (out/(p.stem+'.txt')).write_text(t,encoding='utf-8')
    item={'file':p.as_posix(),'pages':len(d),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'doi_first_pages':sorted(set(re.findall(r'10\.\d{4,9}/[^\s]+',' '.join(pages[:2]))))}
    records.append(item)
    print(json.dumps(item,ensure_ascii=False));print('FIRST PAGE:',pages[0][:1800].replace('\n',' '))
(out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
