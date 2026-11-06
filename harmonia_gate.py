"""harmonia_gate.py — Boy Scouts gate for HarmoniaSynth.mark_trained().

mark_trained() flips the fingerprint on a bare call; nothing stops a human (or a
script) from claiming 'trained' with no weights. This gate is the only sanctioned
way to flip it. It refuses unless ALL of these hold, and appends an audit event.

  1. checkpoint file exists, is non-trivial, and its sha256 is recorded
  2. data manifest: every recording has rights in {"own","written"} + a consent
     doc sha256 + a speaker id; zero entries with any other rights value
  3. eval report: held-out latent L1 present, finite, and not worse than the
     declared baseline (predicting a zero latent); canary_pass is True
  4. report/manifest are bound to THIS checkpoint hash (no reusing old evidence)
Run in a checkout the trained model's author does not control if you want the
'oracle cannot see the suite' property from the Boy Scouts spec.
"""
import hashlib, json, math, time
from pathlib import Path

class GateRefused(Exception): pass

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def attest_trained(model, ckpt, manifest, eval_report, audit_log="harmonia_audit.jsonl", min_bytes=1_000_000):
    ckpt, manifest, eval_report = Path(ckpt), Path(manifest), Path(eval_report)
    problems = []
    if not ckpt.is_file() or ckpt.stat().st_size < min_bytes: problems.append("checkpoint missing or implausibly small")
    ck = _sha(ckpt) if ckpt.is_file() else None
    try: man = json.loads(manifest.read_text())
    except Exception as e: man = None; problems.append(f"manifest unreadable: {e}")
    try: ev = json.loads(eval_report.read_text())
    except Exception as e: ev = None; problems.append(f"eval report unreadable: {e}")
    if man is not None:
        recs = man.get("recordings", [])
        if not recs: problems.append("manifest has no recordings")
        for i, r in enumerate(recs):
            if r.get("rights") not in ("own", "written"): problems.append(f"recording {i}: rights must be own|written")
            if not r.get("consent_doc_sha256") or not r.get("speaker_id"): problems.append(f"recording {i}: missing consent hash or speaker id")
        if man.get("checkpoint_sha256") != ck: problems.append("manifest not bound to this checkpoint hash")
    if ev is not None:
        l1, base = ev.get("heldout_latent_l1"), ev.get("baseline_zero_latent_l1")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (l1, base)): problems.append("eval L1 values missing or non-finite")
        elif l1 >= base: problems.append("held-out L1 not better than zero-latent baseline")
        if ev.get("canary_pass") is not True: problems.append("canary_pass is not True")
        if ev.get("checkpoint_sha256") != ck: problems.append("eval report not bound to this checkpoint hash")
    ok = not problems
    event = {"event": "trained.attested" if ok else "trained.refused", "ts_unix": time.time(),
             "checkpoint_sha256": ck, "problems": problems}
    with open(audit_log, "a") as f: f.write(json.dumps(event) + "\n")
    if not ok: raise GateRefused("; ".join(problems))
    model.mark_trained()
    return event


def check_manifest_rights(manifest, require_files=True):
    """Refuse-to-train check. manifest: path to JSON {"recordings":[{"path","rights","consent_doc_sha256","speaker_id"}]}.
    Returns the parsed manifest, or raises GateRefused listing every problem."""
    problems = []
    try: man = json.loads(Path(manifest).read_text())
    except Exception as e: raise GateRefused(f"manifest unreadable: {e}")
    recs = man.get("recordings", [])
    if not recs: problems.append("manifest has no recordings")
    for i, r in enumerate(recs):
        if r.get("rights") not in ("own", "written"): problems.append(f"recording {i}: rights must be own|written")
        if not r.get("consent_doc_sha256") or not r.get("speaker_id"): problems.append(f"recording {i}: missing consent hash or speaker id")
        if require_files and not (r.get("path") and Path(r["path"]).is_file()): problems.append(f"recording {i}: file missing")
    if problems: raise GateRefused("; ".join(problems))
    return man
