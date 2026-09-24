import json
d = json.load(open('B_huynh.json', encoding='utf-8'))
for it in d['message']['items'][:5]:
    title = it.get('title',['?'])
    title = title[0] if title else '?'
    print(it.get('DOI'), '|', title[:90])
