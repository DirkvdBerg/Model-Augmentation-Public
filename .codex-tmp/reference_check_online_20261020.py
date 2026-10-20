from pathlib import Path
import requests,json,re,concurrent.futures
import pymupdf
out=Path('.codex-tmp/reference-check-20261020')
dois=['10.1016/j.automatica.2023.111210','10.1016/j.conengprac.2012.10.010','10.1109/70.56655','10.1016/j.ifacsc.2026.100376','10.1038/s42254-021-00314-5','10.1080/00207179.2017.1308557','10.1016/j.sysconle.2021.105030','10.1017/dce.2024.8','10.1016/j.mechatronics.2007.04.004','10.1007/978-3-642-13812-6','10.1145/3514228','10.1016/S0005-1098(99)00022-9']
def crossref(doi):
    try:
        r=requests.get('https://api.crossref.org/works/'+doi,timeout=35);r.raise_for_status();m=r.json()['message']
        (out/('crossref-'+doi.replace('/','_')+'.json')).write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
        return {k:m.get(k) for k in ['DOI','title','author','container-title','volume','issue','page','article-number','published','published-print','published-online','link']}
    except Exception as e:return {'doi':doi,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
    for item in ex.map(crossref,dois):print(json.dumps(item,ensure_ascii=False))
urls={
 'official-gyorok2026obc':'https://pure.tue.nl/ws/files/381189611/1-s2.0-S2468601826000167-main.pdf',
 'official-gyorok2025l4dc':'https://raw.githubusercontent.com/mlresearch/v283/main/assets/gyorok25a/gyorok25a.pdf',
 'official-drenth-arxiv':'https://arxiv.org/pdf/2605.12203',
 'official-venkatraman2017psd':'https://proceedings.neurips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf',
}
def inspect(item):
    name,url=item
    try:
        r=requests.get(url,timeout=35);r.raise_for_status()
        d=pymupdf.open(stream=r.content,filetype='pdf')
        # Inspect online bytes; retain extracted evidence only, never save a PDF.
        text='\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+p.get_text(sort=True) for i,p in enumerate(d))
        (out/(name+'.txt')).write_text(text,encoding='utf-8')
        return f'{name}: {len(d)} pages; '+text[:450].replace('\n',' ')
    except Exception as e:return name+': '+str(e)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for item in ex.map(inspect,urls.items()):print(item)
