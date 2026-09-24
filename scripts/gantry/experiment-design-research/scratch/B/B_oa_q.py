import sys,json,urllib.request,urllib.parse
q=sys.argv[1]
url="https://api.openalex.org/works?"+q+"&per-page=25&select=id,doi,title,publication_year,primary_location,cited_by_count"
d=json.load(urllib.request.urlopen(url)); assert 'error' not in d, d
print('COUNT',d['meta']['count'])
for w in d['results']:
    src=((w.get('primary_location') or {}).get('source') or {}).get('display_name')
    print(w['publication_year'],'|',(w['title'] or '')[:110],'|',str(src)[:30],'|',w['doi'],'| cit',w['cited_by_count'])
