import json, sys, urllib.request
ms = sys.argv[1:]
for m in ms:
    try:
        d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}", timeout=20))
        lic = [t for t in d.get("tags", []) if t.startswith("license")]
        print(f"{m:55s} gated={d.get('gated')!s:7s} {lic} mod={str(d.get('lastModified',''))[:10]} dl={d.get('downloads')}")
    except Exception as e:
        print(f"{m:55s} ERR {e}")
