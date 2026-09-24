import sys,re
t=open(sys.argv[1],encoding='utf-8').read()
tot=re.search(r'totalResults[^>]*>(\d+)',t)
print("TOTAL",tot.group(1) if tot else None)
for e in re.findall(r'<entry>(.*?)</entry>',t,re.S):
    i=re.search(r'<id>([^<]+)',e).group(1); ti=" ".join(re.search(r'<title>([^<]+)',e,re.S).group(1).split())
    au=", ".join(re.findall(r'<name>([^<]+)',e)[:4]); pub=re.search(r'<published>(\d{4})',e).group(1)
    print(f"{pub} | {i} | {ti} | {au}")
