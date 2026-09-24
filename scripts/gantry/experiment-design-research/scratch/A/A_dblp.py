import sys,json
d=json.load(open(sys.argv[1],encoding='utf-8'))
r=d['result']['hits']; print("TOTAL",r.get('@total'))
for h in r.get('hit',[]):
    i=h['info']; au=i.get('authors',{}).get('author',[]); au=au if isinstance(au,list) else [au]
    print(f"{i.get('year')} [{str(i.get('venue'))[:22]}] {i.get('title','')[:110]} | {', '.join(a['text'] for a in au[:3])} | {i.get('ee')}")
