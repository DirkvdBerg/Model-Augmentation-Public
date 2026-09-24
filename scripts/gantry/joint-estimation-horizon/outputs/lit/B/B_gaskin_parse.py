import json
d = json.load(open('B_gaskin.json', encoding='utf-8'))
for it in d['message']['items'][:5]:
    title = it.get('title',['?'])
    title = title[0] if title else '?'
    auth = ", ".join(a.get('family','') for a in it.get('author',[])[:4])
    print(it.get('DOI'), '|', it.get('issued',{}).get('date-parts',[[None]])[0][0], '|', auth, '|', title[:90])
