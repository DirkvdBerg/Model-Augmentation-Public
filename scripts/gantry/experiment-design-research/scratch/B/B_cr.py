import sys,json,urllib.request,urllib.parse,time
qs=[l.strip() for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
for q in qs:
    url="https://api.crossref.org/works?rows=2&query.bibliographic="+urllib.parse.quote(q)
    try:
        d=json.load(urllib.request.urlopen(url,timeout=30))
    except Exception as e:
        print('ERR',q,e); continue
    print('>>',q)
    for it in d['message']['items'][:2]:
        au=", ".join((a.get('family') or '')+' '+(a.get('given') or '')[:1] for a in it.get('author',[])[:5])
        yr=(it.get('issued',{}).get('date-parts') or [[None]])[0][0]
        print('   ',yr,'|',(it.get('title') or [''])[0][:100],'|',(it.get('container-title') or [''])[0][:50],'|',it.get('volume'),it.get('page'),'|',it.get('DOI'),'|',au)
    time.sleep(1)
