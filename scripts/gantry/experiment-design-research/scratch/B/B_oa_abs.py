import sys,json,urllib.request
url="https://api.openalex.org/works?filter=doi:"+sys.argv[1]+"&per-page=25&select=id,doi,title,publication_year,authorships,abstract_inverted_index,primary_location,open_access,locations"
d=json.load(urllib.request.urlopen(url)); assert 'error' not in d, d
for w in d['results']:
    inv=w.get('abstract_inverted_index') or {}
    pos=sorted((p,wd) for wd,ps in inv.items() for p in ps)
    ab=" ".join(wd for p,wd in pos)
    au=", ".join(a['author']['display_name'] for a in w['authorships'][:6])
    src=((w.get('primary_location') or {}).get('source') or {}).get('display_name')
    print('##',w['publication_year'],w['title']); print('  ',au,'|',src,'|',w['doi'],'|',w['open_access'].get('oa_status'))
    for L in w['locations']:
        if L.get('is_oa'): print('   [OA]',((L.get('source') or {}).get('display_name')),L.get('pdf_url'),L.get('landing_page_url'))
    print('   ABSTRACT:',ab[:1500])
