"""train_synth.py — train the HarmoniaSynth head (phonemes + F0 -> codec latent).

PREREQUISITES (see TRAINING.md): codec weights from train_codec_ae.py; per-recording feature files
from your aligner + F0 tracker. STATUS: syntax-checked only, never run (authoring sandbox had no torch).

Feature file (<name>.feat.npz) for each recording, frame rate 120 Hz (HOP=200 @ 24 kHz):
    phon   int   [T]   phoneme IDs (< n_phonemes, default 128)
    f0     float [T]   Hz, 0 where unvoiced
    voiced float [T]   1/0
    loud   float [T]   log-RMS (or similar), per frame
with T >= len(wav)//200. Manifest rules: docs/DATA_MANIFEST.md (own|written rights only).

Writes <out>/harmonia_synth.pt, <out>/synth_eval.json, <out>/manifest.bound.json. It does NOT mark the model
trained: set canary_pass after running the harness canaries, then call harmonia_gate.attest_trained().
"""
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from harmonia_gate import check_manifest_rights, GateRefused, _sha
from harmonia_synth import build_singing_synth, train_step, HOP
from train_codec_ae import load_wav

FRAMES = 480                                  # 4 s crops

def load_item(r):
    f = Path(r.get("features", ""))
    if not f.is_file(): raise GateRefused(f"features file missing for {r['path']}")
    d = np.load(f); wav = load_wav(r["path"]); T = len(wav) // HOP
    feats = {"phon": d["phon"].astype(np.int64), "f0": d["f0"].astype(np.float32),
             "voiced": d["voiced"].astype(np.float32), "loud": d["loud"].astype(np.float32)}
    for k, v in feats.items():
        if len(v) < T: raise GateRefused(f"{r['path']}: feature '{k}' shorter than audio ({len(v)} < {T})")
    return wav[: T * HOP], {k: v[:T] for k, v in feats.items()}, T

def batch_of(items, B, rng, dev):
    cols = {k: [] for k in ("wav", "phon", "f0", "voiced", "loud")}
    for _ in range(B):
        wav, ft, T = rng.choice(items)
        n = min(FRAMES, T); t0 = rng.randrange(0, T - n + 1); pad = FRAMES - n
        cols["wav"].append(np.pad(wav[t0 * HOP:(t0 + n) * HOP], (0, pad * HOP)))
        for k in ("phon", "f0", "voiced", "loud"):
            cols[k].append(np.pad(ft[k][t0:t0 + n], (0, pad)))
    t = lambda k, dt: torch.from_numpy(np.stack(cols[k])).to(dev, dt)
    return {"wav": t("wav", torch.float32)[:, None], "phon": t("phon", torch.long), "f0": t("f0", torch.float32),
            "voiced": t("voiced", torch.float32), "loud": t("loud", torch.float32)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--codec", required=True, help="dir with codec_encoder.pt/codec_decoder.pt")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    man = check_manifest_rights(a.manifest)
    recs = man["recordings"]
    if len(recs) < 2: raise SystemExit("need >=2 recordings (the last is held out)")
    items = [load_item(r) for r in recs[:-1]]; held = load_item(recs[-1])
    c = Path(a.codec)
    synth, enc = build_singing_synth(causal=False, device=a.device, encoder_ckpt=str(c / "codec_encoder.pt"), decoder_ckpt=str(c / "codec_decoder.pt"))
    opt = torch.optim.AdamW([p for p in synth.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps); rng = random.Random(0); t0 = time.time()
    synth.train()
    for step in range(1, a.steps + 1):
        loss = train_step(synth, enc, batch_of(items, a.batch, rng, a.device), opt); sched.step()
        if step % 500 == 0: print(f"step {step} latent-L1 {loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
    synth.eval(); wav, ft, T = held; T = min(T, 3600)
    with torch.no_grad():
        x = torch.from_numpy(wav[: T * HOP])[None, None].to(a.device)
        z = enc(x); tt = lambda k, dt: torch.from_numpy(ft[k][:T])[None].to(a.device, dt)
        zh = synth.predict_latent(tt("phon", torch.long), tt("f0", torch.float32), tt("voiced", torch.float32), tt("loud", torch.float32))
        n = min(z.shape[-1], zh.shape[-1]); l1 = float((zh[..., :n] - z[..., :n]).abs().mean()); base = float(z[..., :n].abs().mean())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(synth.state_dict(), out / "harmonia_synth.pt"); h = _sha(out / "harmonia_synth.pt")
    (out / "synth_eval.json").write_text(json.dumps({"checkpoint_sha256": h, "heldout_latent_l1": l1, "baseline_zero_latent_l1": base,
        "canary_pass": None, "note": "set canary_pass=true only after running the harness canaries; then call harmonia_gate.attest_trained()"}, indent=1))
    (out / "manifest.bound.json").write_text(json.dumps({**man, "checkpoint_sha256": h}, indent=1))
    print(f"heldout L1 {l1:.4f} vs zero-latent baseline {base:.4f}; wrote {out}")

if __name__ == "__main__": main()
