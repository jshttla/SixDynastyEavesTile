import sys, json, urllib.request
for m in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}", timeout=20))
        print(m, "gated=", d.get("gated"), [s["rfilename"] for s in d.get("siblings", [])][:30])
    except Exception as e:
        print(m, "ERR", e)
