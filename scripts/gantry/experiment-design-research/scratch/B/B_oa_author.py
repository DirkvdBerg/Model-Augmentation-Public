import sys,json,urllib.request,urllib.parse,time
aid=sys.argv[1]; out=sys.argv[2]
works=[]; cursor='*'; n=0
while cursor and n<3:
    url=f"https://api.openalex.org/works?filter=author.id:{aid}&per-page=200&cursor={urllib.parse.quote(cursor)}&select=id,doi,title,publication_year,primary_location,type"
    d=json.load(urllib.request.urlopen(url)); n+=1
    assert 'error' not in d, d
    works+=d['results']; cursor=d['meta'].get('next_cursor')
    print('page',n,'got',len(d['results']),'total',d['meta']['count'])
    if len(d['results'])<200: break
    time.sleep(1)
json.dump(works,open(out,'w',encoding='utf-8'))
print('saved',len(works),'queries',n)
