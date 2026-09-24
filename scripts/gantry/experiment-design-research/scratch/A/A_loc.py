import sys,json
d=json.load(open(sys.argv[1],encoding='utf-8'))
assert 'error' not in d, d
print(d.get('title'),'|',d.get('open_access',{}).get('oa_status'),'| cited',d.get('cited_by_count'),'|',d.get('id'))
for L in d.get('locations',[]):
    s=(L.get('source') or {})
    print(f"  [{'OA' if L.get('is_oa') else '  '}] {str(s.get('display_name'))[:32]:32s} pdf={L.get('pdf_url')} land={L.get('landing_page_url')}")
