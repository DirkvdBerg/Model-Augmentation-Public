import sys,json
d=json.load(open(sys.argv[1],encoding='utf-8'))
for it in d['message']['items']:
    au=", ".join((a.get('given','')+' '+a.get('family','')) for a in it.get('author',[])[:5])
    print(it.get('title'),'|',au,'|',it.get('container-title'),'|',it.get('issued',{}).get('date-parts'),'|',it.get('volume'),it.get('issue'),it.get('page'),'|',it.get('DOI'))
