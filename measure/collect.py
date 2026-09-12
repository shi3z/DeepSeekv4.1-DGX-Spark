"""Turn the drive.py logs in results/ into one comparison table.

Every row carries what makes the tok/s number mean something: the acceptance length it was
measured at (their own sweep moves 20 -> 30 tok/s on acceptance alone), how much of the routed
expert set the configuration can actually reach, and whether free generation degenerated.
"""
import glob, json, os, re, sys

rows = []
for path in sorted(glob.glob("results/*.log")):
    label = os.path.basename(path)[:-4]
    r = {"run": label}
    for line in open(path, errors="replace"):
        if line.startswith("CONFIG "):
            r["config"] = json.loads(line[7:])
        elif line.startswith("RESULT "):
            m = re.search(r"= ([\d.]+) tok/s .*acceptance ([\d.None]+) .*hit ([\d.]+) .*NVMe ([\d.]+) GB", line)
            if m:
                r["tok_s"], r["accept"], r["hit"], r["nvme_gb"] = m.group(1), m.group(2), m.group(3), m.group(4)
        elif line.startswith("STATS "):
            r["stats"] = json.loads(line[6:])
        elif line.startswith("QUALITY "):
            r["quality"] = json.loads(line[8:line.rindex("}") + 1])
        elif line.startswith("FREEGEN "):
            m = re.search(r"(\d+) tok, ([\d.]+) tok/s, distinct-token ratio ([\d.]+)", line)
            if m:
                r["free_tok"], r["free_tok_s"], r["distinct"] = m.group(1), m.group(2), m.group(3)
                r["degenerate"] = float(m.group(3)) < 0.25
        elif "every routed expert is resident" in line:
            r["all_resident"] = True
        elif "tiered arena:" in line:
            r["plan"] = line.split("tiered arena:", 1)[1].strip()
    rows.append(r)

hdr = f"{'run':26s} {'tok/s':>7s} {'accept':>7s} {'reach':>7s} {'code NLL':>9s} {'gen NLL':>8s} {'distinct':>9s}"
print(hdr); print("-" * len(hdr))
for r in rows:
    q = r.get("quality") or {}
    cfg = r.get("config") or {}
    reach = "100%" if r.get("all_resident") else (
        f"{float(cfg.get('prune_keep') or 1.0) * 100:.0f}%" if cfg.get("prune_keep") else "stream")
    print(f"{r['run']:26s} {r.get('tok_s','-'):>7s} {r.get('accept','-'):>7s} {reach:>7s} "
          f"{q.get('coding',{}).get('nll','-'):>9} {q.get('general',{}).get('nll','-'):>8} "
          f"{r.get('distinct','-'):>9s}{'  DEGENERATE' if r.get('degenerate') else ''}")
print()
for r in rows:
    if r.get("plan"):
        print(f"{r['run']}: {r['plan']}")
json.dump(rows, open("results/summary.json", "w"), indent=1)
print("\n-> results/summary.json")
