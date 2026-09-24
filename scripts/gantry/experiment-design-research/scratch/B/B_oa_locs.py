import sys,json,urllib.request
ids=sys.argv[1]
url=f"https://api.openalex.org/works?filter=ids.openalex:{ids}&per-page=50&select=id,title,doi,locations,open_access,authorships,publication_year"
d=json.load(urllib.request.urlopen(url)); assert 'error' not in d, d
json.dump(d,open('locs.json','w',encoding='utf-8'))
for w in d['results']:
    au=", ".join(a['author']['display_name'] for a in w['authorships'][:6])
    print('##',w['id'].split('/')[-1],w['publication_year'],w['title'][:100]); print('   ',au,'|',w['doi'],'|',w['open_access'].get('oa_status'))
    for L in w['locations']:
        s=(L.get('source') or {}).get('display_name')
        if L.get('is_oa') or L.get('pdf_url'): print('    [OA]' if L.get('is_oa') else '    [  ]',str(s)[:30],'pdf=',L.get('pdf_url'),'lp=',L.get('landing_page_url'))
