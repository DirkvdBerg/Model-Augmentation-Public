import sys,json
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert 'error' not in d, d
print("COUNT",d.get('meta',{}).get('count'))
for w in d.get('results',[]):
    au=", ".join(a['author']['display_name'] for a in w.get('authorships',[])[:4])
    src=((w.get('primary_location') or {}).get('source') or {}).get('display_name')
    print(f"{w.get('publication_year')} | {w.get('title')} | {au} | {src} | {w.get('doi')} | {w.get('id')} | cited {w.get('cited_by_count')} | oa {w.get('open_access',{}).get('oa_status')}")
