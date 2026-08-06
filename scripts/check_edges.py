import json
g = json.load(open(r"d:\proyectoBolt\governance-agent\nomenclador\nomenclador.json", encoding="utf-8"))
print("Keys:", list(g.keys()))
links = g.get("links", [])
equiv = [e for e in links if e.get("type") == "EQUIVALE_A"]
print(f"Total links: {len(links)}")
print(f"EQUIVALE_A links: {len(equiv)}")
for e in equiv[:5]:
    print(f"  {e.get('source','?')} -> {e.get('target','?')} conf={e.get('confidence','?')}")
