import sys,json,re
w=json.load(open(sys.argv[1],encoding='utf-8'))
pat=re.compile(sys.argv[2],re.I)
for x in sorted(w,key=lambda x:x.get('publication_year') or 0):
    t=x.get('title') or ''
    if pat.search(t):
        src=((x.get('primary_location') or {}).get('source') or {}).get('display_name')
        print(x.get('publication_year'),'|',t[:120],'|',str(src)[:40],'|',x.get('doi'),'|',x['id'].split('/')[-1])
