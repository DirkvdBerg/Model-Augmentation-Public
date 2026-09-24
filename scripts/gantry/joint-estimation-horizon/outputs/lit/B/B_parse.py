import json, sys
for fn in ["B_borkar.json","B_konda.json","B_varpro.json","B_vonstosch.json","B_bradley.json"]:
    print(f"=== {fn} ===")
    try:
        d = json.load(open(fn, encoding="utf-8"))
    except Exception as e:
        print("PARSE ERROR", e); continue
    items = d.get("message", {}).get("items", [])
    for it in items[:5]:
        title = it.get("title", ["?"])
        title = title[0] if title else "?"
        authors = it.get("author", [])
        au = ", ".join([f"{a.get('family','')}" for a in authors[:4]])
        year = it.get("issued",{}).get("date-parts",[[None]])[0][0]
        doi = it.get("DOI")
        venue = it.get("container-title", ["?"])
        venue = venue[0] if venue else "?"
        print(f"  {year} | {au} | {title[:80]} | {venue[:40]} | DOI {doi}")
