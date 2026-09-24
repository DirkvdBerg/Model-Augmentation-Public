import sys,re
t=open(sys.argv[1],encoding='utf-8').read()
m=re.search(r'<opensearch:totalResults[^>]*>(\d+)',t); print('TOTAL',m.group(1) if m else None)
for e in t.split('<entry>')[1:]:
    ti=re.search(r'<title>(.*?)</title>',e,re.S).group(1); i=re.search(r'<id>(.*?)</id>',e).group(1)
    au=re.findall(r'<name>(.*?)</name>',e)[:4]; y=re.search(r'<published>(\d{4})',e).group(1)
    print(y,i.split('/')[-1],'|',' '.join(ti.split())[:110],'|',', '.join(au))
