#!/usr/bin/env python3
"""Report which HarmoniaCodec Music components still need training, weights, or a code fix.

Pure standard library (no torch). Reads MODEL_STATUS.json, inspects files on disk.
Statuses: NEEDS_TRAINING | SCRIPT_MISSING | WEIGHTS_PRESENT_UNVERIFIED | WEIGHTS_PRESENT |
          FAILED_GATE | MISSING_EXTERNAL | READY | BLOCKED
File presence is NOT proof of quality; WEIGHTS_PRESENT means the component's gate file says it passed.
Exit code 1 if any component required for --target is not READY/WEIGHTS_PRESENT (use --no-fail in CI).
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK = {"READY", "WEIGHTS_PRESENT"}

def _gate(root, c):
    gf = c.get("gate_file")
    if not gf: return None
    p = root / gf
    if not p.is_file(): return "unverified"
    try: v = json.loads(p.read_text()).get(c.get("gate_key"))
    except Exception: return "failed"
    return "ok" if v is True else ("unverified" if v is None else "failed")

def evaluate(reg, root=ROOT):
    rows = []
    for c in reg["components"]:
        root = Path(root); outs = [root / o for o in c.get("outputs", [])]
        present = bool(outs) and all(o.exists() for o in outs)
        script_ok = (root / c["script"]).is_file() if c.get("script") else None
        k = c["kind"]; action = ""
        if k == "code_fix":
            st = "READY" if c.get("fixed") else "BLOCKED"; action = c.get("note", "")
        elif k == "external":
            st = "READY" if present else "MISSING_EXTERNAL"; action = "" if present else c.get("fetch", "")
        else:
            if c.get("blocked_by"): st, action = "BLOCKED", c["blocked_by"]
            elif not script_ok: st, action = "SCRIPT_MISSING", "write/restore " + str(c.get("script"))
            elif not present: st, action = "NEEDS_TRAINING", c.get("command", "")
            else:
                g = _gate(root, c)
                st = {"ok": "WEIGHTS_PRESENT", "failed": "FAILED_GATE"}.get(g, "WEIGHTS_PRESENT_UNVERIFIED")
                if st != "WEIGHTS_PRESENT": action = "run its evaluation/gate; see TRAINING.md"
        rows.append({"id": c["id"], "kind": k, "status": st, "required_for": c.get("required_for", []),
                     "depends_on": c.get("depends_on", []), "action": action})
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--registry", default=str(ROOT / "MODEL_STATUS.json")); ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--target", default="singing"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-fail", action="store_true", help="always exit 0 (CI)")
    a = ap.parse_args(argv)
    rows = evaluate(json.loads(Path(a.registry).read_text()), a.root)
    missing = [r["id"] for r in rows if a.target in r["required_for"] and r["status"] not in OK]
    if a.json: print(json.dumps({"target": a.target, "rows": rows, "not_ready": missing}, indent=1))
    else:
        print(f"HarmoniaCodec Music - model status (target: {a.target})\n")
        for r in rows:
            req = "*" if a.target in r["required_for"] else " "
            print(f"{req} {r['id']:<20} {r['kind']:<10} {r['status']:<28} {r['action']}")
        print("\n* = required for target. " + ("NOT READY: " + ", ".join(missing) if missing else "All required components present."))
    return 0 if (not missing or a.no_fail) else 1

if __name__ == "__main__": sys.exit(main())
