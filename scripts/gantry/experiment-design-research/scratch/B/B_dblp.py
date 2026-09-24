import sys,json
d=json.load(open(sys.argv[1],encoding='utf-8'))
print('TOTAL',d['result']['hits'].get('@total'))
for h in d['result']['hits'].get('hit',[]):
    i=h['info']; a=i.get('authors',{}).get('author',[]); a=a if isinstance(a,list) else [a]
    print(i.get('year'),'|',str(i.get('venue'))[:22],'|',i.get('title','')[:100],'|',', '.join(x['text'] for x in a[:4]),'|',i.get('ee'))
