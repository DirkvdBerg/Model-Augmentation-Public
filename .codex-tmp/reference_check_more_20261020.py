from pathlib import Path
import requests,json,hashlib
import pymupdf
from urllib.parse import quote
out=Path('.codex-tmp/reference-check-20261020')
for doi in ['10.1016/j.automatica.2023.111210','10.1109/70.56655','10.1038/s42254-021-00314-5','10.1016/j.sysconle.2021.105030','10.1145/3514228']:
    try:
        r=requests.get('https://api.crossref.org/works/'+quote(doi,safe=''),headers={'User-Agent':'ReferenceAudit/1.0','Accept':'application/json'},timeout=25);r.raise_for_status();m=r.json()['message']
        (out/('crossref-'+doi.replace('/','_')+'.json')).write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:m.get(k) for k in ['DOI','title','author','container-title','volume','issue','page','article-number','published','published-print','published-online']},ensure_ascii=False))
    except Exception as e:print(doi,str(e))
urls={
 'official-garcia-publisher':'https://api.elsevier.com/content/article/pii/S0967066112002328?httpAccept=text/xml',
 'official-ovchinnikov-publisher':'https://api.elsevier.com/content/article/pii/S0167691121001602?httpAccept=text/xml',
 'official-drenth-author':'https://research.tue.nl/en/publications/?search=Efficient+Learning+of+Affine',
 'official-downey-metadata':'https://proceedings.neurips.cc/paper_files/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Metadata.json',
 'official-hefny-metadata':'https://proceedings.neurips.cc/paper_files/paper/2015/file/9a3d458322d70046f63dfd8b0153ece4-Metadata.json',
 'official-venkatraman-metadata':'https://proceedings.neurips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Metadata.json',
}
for name,url in urls.items():
    try:
        r=requests.get(url,timeout=30);r.raise_for_status()
        (out/(name+'.txt')).write_text(r.text,encoding='utf-8')
        if name.endswith('metadata'):
            m=r.json();print(name,json.dumps({k:v for k,v in m.items() if k!='full_text' and k!='abstract'},ensure_ascii=False)[:2000])
        else:print(name,r.status_code,r.text[:100])
    except Exception as e:print(name,str(e))
# Other local copies: inspect front matter, retain only text.
for name,p in {
 'alternative-hefny':'literature/added-states-training/hefny2015_supervised-learning-dynamical-system-learning.pdf',
 'alternative-gyorokobc':'literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf',
 'alternative-garcia':'literature/gantry/garcia2013_gantry-decoupling-control.pdf',
}.items():
    d=pymupdf.open(p)
    text='\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+page.get_text(sort=True) for i,page in enumerate(d))
    (out/(name+'.txt')).write_text(text,encoding='utf-8')
    print(name,len(d),hashlib.sha256(Path(p).read_bytes()).hexdigest(),text[:250].replace('\n',' '))
