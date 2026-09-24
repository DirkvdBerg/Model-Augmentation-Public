import json
d = json.load(open('B_gaskin_oa.json', encoding='utf-8'))
assert 'error' not in d, d
print(d.get('title'), '|', d.get('open_access',{}).get('oa_status'))
for L in d.get('locations',[]):
    s = (L.get('source') or {})
    print(f"  [{'OA' if L.get('is_oa') else '  '}] {str(s.get('display_name'))[:32]:32s} pdf={L.get('pdf_url')} landing={L.get('landing_page_url')}")
