from pathlib import Path
import re,json,hashlib
import pymupdf
root=Path.cwd()
out=root/'.codex-tmp/reference-check-20261020'
out.mkdir(exist_ok=True)
base=root/'Thesis-writeup/Writing'
files=sorted((base/'sections').glob('*.tex'))+[base/'main.tex']+sorted((base/'frontmatter').rglob('*.tex'))
records=[]; hashes={}
def group(s,i,op='{',cl='}'):
    depth=1;j=i+1
    while j<len(s) and depth:
        if s[j]==op and (j==0 or s[j-1]!='\\'):depth+=1
        if s[j]==cl and (j==0 or s[j-1]!='\\'):depth-=1
        j+=1
    return j
for p in files:
    raw=p.read_text(encoding='utf-8-sig'); hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    s=re.sub(r'(?<!\\)%[^\n]*',lambda m:' '*len(m[0]),raw)
    spans=[]
    for m in re.finditer(r'\\(delmv|del|todo|revnote|chg|mv)\b(?:\[[^\]]*\])?\s*',s):
        i=m.end();n=2 if m[1] in ('mv','delmv','chg') else 1
        for arg in range(n):
            while i<len(s) and s[i].isspace():i+=1
            if i>=len(s) or s[i]!='{':break
            j=group(s,i)
            if m[1] not in ('mv','chg') or (m[1]=='chg' and arg==0):spans.append((i,j,m[1]))
            i=j
    wips=[(m.start(),s.find('\\end{wip}',m.end())) for m in re.finditer(r'\\begin\{wip\}',s)]
    for m in re.finditer(r'\\(u?cite)\s*(?:\[([^\]]*)\])?\s*\{([^{}]*)\}',s):
        flags=[flag for a,b,flag in spans if a<=m.start()<b]
        if any(a<m.start()<b for a,b in wips):flags.append('wip')
        for key in m[3].split(','):
            records.append(dict(key=key.strip(),marker=m[1],pinpoint=m[2] or '',file=str(p.relative_to(base)).replace('\\','/'),line=s.count('\n',0,m.start())+1,flags=flags))
(out/'citations.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(out/'hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print('CITATIONS',len(records),'KEYS',len(set(r['key'] for r in records)))
for k in sorted(set(r['key'] for r in records)):
    rr=[r for r in records if r['key']==k]
    print(k,':','; '.join(f"{r['file']}:{r['line']} [{r['pinpoint']}] {','.join(r['flags']) or 'body'}" for r in rr))
pdfs=list((root/'literature/thesis-referenced').glob('*.pdf'))
pdfs+=list((root/'literature/lpv-lfr').glob('*drenth*pdf'))
manifest=[]
for p in pdfs:
    d=pymupdf.open(p)
    pages=[page.get_text(sort=True) for page in d]
    name=p.stem if p.parent.name=='thesis-referenced' else 'alternative-'+p.stem
    (out/(name+'.txt')).write_text('\n'.join(f'\n=== PDF PAGE {i+1} ===\n'+t for i,t in enumerate(pages)),encoding='utf-8')
    manifest.append(dict(file=str(p.relative_to(root)),pages=len(d),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    print('\nPDF',str(p.relative_to(root)),len(d),'pages\n',pages[0][:1900].replace('\n',' '))
(out/'pdfs.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
