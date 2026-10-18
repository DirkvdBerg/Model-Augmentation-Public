"""Read-only thesis inventory; outputs are confined to this audit directory."""
from pathlib import Path
import re,json,csv,hashlib,subprocess,collections
BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ROOT=OUT/'snapshot/Writing'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def uncomment(s): return re.sub(r'(?<!\\)%[^\n]*',lambda m:' '*len(m[0]),s)
def group(s,p,left='{',right='}'):
    if p>=len(s) or s[p]!=left:return None,p
    start=p+1; depth=1;p+=1
    while p<len(s) and depth:
        if s[p]=='\\':p+=2;continue
        if s[p]==left:depth+=1
        if s[p]==right:depth-=1
        p+=1
    return s[start:p-1],p
def args(s,p,n=1):
    while p<len(s) and s[p].isspace():p+=1
    opt=''
    if p<len(s) and s[p]=='[':opt,p=group(s,p,'[',']')
    a=[]
    for _ in range(n):
        while p<len(s) and s[p].isspace():p+=1
        start=p;v,p=group(s,p)
        if v is None:break
        a.append((start,p,v))
    return opt,a,p
files={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8-sig',errors='replace') for p in ROOT.rglob('*.tex')}
graph={}; reachable=set()
for f,s in files.items():
    graph[f]=[]
    for m in re.finditer(r'\\(?:input|include)\s*\{([^}]+)\}',uncomment(s)):
        t=m[1];t=t if Path(t).suffix else t+'.tex'
        graph[f].append({'target':t,'line':s.count('\n',0,m.start())+1})
def visit(f):
    if f in reachable:return
    reachable.add(f)
    for e in graph.get(f,[]):visit(e['target'])
visit('main.tex')
rows=[]
for f,s in sorted(files.items()):
    clean=uncomment(s);spans=[]; stack=[]
    for m in re.finditer(r'\\(?:ifdraft|ifdefined|iffalse|else|fi)\b|\\(?:begin|end)\{(?:wip|outline|comment)\}',clean):
        tok=m[0]
        if tok in ('\\ifdraft','\\ifdefined','\\iffalse') or tok.startswith('\\begin'):
            stack.append((m.end(),tok))
        elif tok=='\\else' and stack:
            st,t=stack.pop();spans.append((st,m.start(),t));stack.append((m.end(),'else:'+t))
        elif stack:
            st,t=stack.pop();spans.append((st,m.start(),t))
    for m in re.finditer(r'\\(todo|note|revnote|del|chg)\b',clean):
        _,a,_=args(clean,m.end(),2 if m[1]=='chg' else 1)
        for i,(st,en,_) in enumerate(a):spans.append((st,en,m[1]+(' old' if m[1]=='chg' and i==0 else ' new' if m[1]=='chg' else '')))
    for m in re.finditer(r'\\([A-Za-z]*cite[A-Za-z]*|nocite)\*?',s):
        cmd=m[1]
        if cmd not in ('cite','ucite','fbcite','nocite','citep','citet','autocite','parencite','textcite','fullcite'):continue
        opt,a,end=args(s,m.end())
        if not a or any('#' in k for k in a[0][2].split(',')):continue
        iscomment=clean[m.start()]==' '
        tags=[t for st,en,t in spans if st<=m.start()<en]
        active=f in reachable and not iscomment and '\\iffalse' not in tags and '\\begin{comment}' not in tags
        review=active and cmd!='ucite' and not any(t in ('todo','note','del','chg old','\\ifdraft','\\begin{wip}') for t in tags)
        line=s.count('\n',0,m.start())+1
        # Full paragraph context, including markup, retained to avoid changing claim scope.
        st=s.rfind('\n\n',0,m.start())+2;en=s.find('\n\n',end);en=len(s) if en<0 else en
        rows.append(dict(id=f'C{len(rows)+1:03}',file=f,line=line,command=cmd,locator=opt,keys=[k.strip() for k in a[0][2].split(',')],reachable=f in reachable,commented=iscomment,tags=tags,main_visible=active,review_visible=review,context=s[st:en],status='NOT YET ASSESSED'))
save('occurrences.json',rows)
with (OUT/'occurrences.csv').open('w',encoding='utf-8-sig',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows({**r,'keys':'; '.join(r['keys']),'tags':'; '.join(r['tags'])} for r in rows)
entries=[];strings=[]
for p in ROOT.rglob('*.bib'):
    s=uncomment(p.read_text(encoding='utf-8-sig')); pos=0
    while m:=re.search(r'@([A-Za-z]+)\s*\{',s[pos:]):
        start=pos+m.start();typ=m[1];body,end=group(s,pos+m.end()-1);pos=end
        if typ.lower() in ('string','preamble','comment'):
            strings.append({'file':p.relative_to(ROOT).as_posix(),'type':typ});continue
        key=body.split(',',1)[0].strip()
        fields={}
        for fm in re.finditer(r'(\w+)\s*=\s*',body):
            k=fm[1].lower();b=fm.end()
            if body[b:b+1]=='{':v,_=group(body,b)
            elif body[b:b+1]=='"':v=body[b+1:body.find('"',b+1)]
            else:v=body[b:].split(',')[0].strip()
            fields[k]=v
        entries.append(dict(file=p.relative_to(ROOT).as_posix(),line=s.count('\n',0,start)+1,type=typ,key=key,fields=fields,cited_main=any(r['main_visible'] and key in r['keys'] for r in rows),cited_review=any(r['review_visible'] and key in r['keys'] for r in rows)))
save('bibliography.json',entries)
save('scope.json',{'root':'main.tex','alternate_root':'review.tex','included_files':sorted(reachable),'unused_tex_files':sorted(set(files)-reachable),'input_graph':graph,'string_and_control_bib_records':strings,'counts':dict(total_occurrences=len(rows),main_visible=sum(r['main_visible'] for r in rows),review_visible=sum(r['review_visible'] for r in rows),bib_entries=len(entries))})
manifest=[]
for p in sorted(set([ROOT/f for f in files]+list(ROOT.rglob('*.bib'))+list((BASE/'literature/thesis-referenced').glob('*.pdf')))):
    manifest.append({'path':p.relative_to(BASE).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
save('input-manifest.json',manifest)
(OUT/'source-text').mkdir(exist_ok=True)
for p in (BASE/'literature/thesis-referenced').glob('*.pdf'):
    subprocess.run(['pdftotext','-layout',str(p),str(OUT/'source-text'/f'{p.stem}.txt')],capture_output=True,check=True)
subprocess.run(['pdftotext','-layout',str(ROOT/'build/main.pdf'),str(OUT/'rendered-main.txt')],capture_output=True,check=True)
print(json.dumps({'counts':json.loads((OUT/'scope.json').read_text())['counts'],'cited_keys':sorted({k for r in rows if r['main_visible'] for k in r['keys']})},indent=2))
