"""train_codec_ae.py — give HarmoniaCodec's encoder/decoder real weights (stage 0).

WHY: HarmoniaSynth regresses onto z = AcousticEncoderNN(wav). With random encoder
weights that target is meaningless, and the decoder has never learned to invert it.
Fix: train encoder+decoder as a plain autoencoder (no VQ; the synth bypasses it) on
rights-cleared audio, THEN freeze both and train the synth head.

STATUS: written without torch in the authoring sandbox; syntax-checked, NOT run.
No discriminator/GAN loss, so expect muffled/buzzy output: judge by ear before
trusting it as the teacher. Not a finished vocoder.
Input: 16-bit PCM mono 24 kHz wavs (convert: ffmpeg -i in.wav -ac 1 -ar 24000 -sample_fmt s16 out.wav).
Usage: python train_codec_ae.py --manifest manifest.json --out ckpt_codec --steps 100000 [--causal]
"""
import argparse, json, random, time, wave
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from harmonia_neural import AcousticEncoderNN, SubBandDecoderNN
from harmonia_gate import check_manifest_rights, _sha

SR, HOP, CROP = 24000, 200, 96000          # 4 s crops, multiple of HOP

def load_wav(path):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise ValueError(f"{path}: need 24 kHz mono 16-bit PCM")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0

def mr_stft(target, recon, ffts=(512, 1024, 2048)):
    tot = 0.0
    for n in ffts:
        win = torch.hann_window(n, device=target.device)
        X = torch.stft(target.squeeze(1), n, n // 4, n, win, return_complex=True).abs() + 1e-7
        Y = torch.stft(recon.squeeze(1), n, n // 4, n, win, return_complex=True).abs() + 1e-7
        tot = tot + (X.log() - Y.log()).abs().mean() + (X - Y).norm() / X.norm()
    return tot / len(ffts)

def crops(arrs, B, rng):
    out = []
    for _ in range(B):
        a = rng.choice(arrs)
        if len(a) < CROP: a = np.pad(a, (0, CROP - len(a)))
        s = rng.randrange(0, len(a) - CROP + 1)
        out.append(a[s:s + CROP])
    return torch.from_numpy(np.stack(out))[:, None]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--causal", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    man = check_manifest_rights(a.manifest)                        # refuses without written rights
    files = [r["path"] for r in man["recordings"]]
    if len(files) < 2: raise SystemExit("need >=2 files (one is held out for evaluation)")
    held, train = load_wav(files[-1]), [load_wav(f) for f in files[:-1]]
    enc = AcousticEncoderNN(dim=128, latent=64, strides=(2, 4, 5, 5), causal=a.causal).to(a.device)
    dec = SubBandDecoderNN(latent=64, dim=128, strides=(5, 5, 4, 2), causal=a.causal).to(a.device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=a.lr, betas=(0.8, 0.99))
    rng = random.Random(0); t0 = time.time()

    def recon(x):
        if a.causal: enc.reset(); dec.reset()                       # stateful convs: reset per independent sequence
        z = enc(x); y = dec(z)[..., : x.shape[-1]]                  # decoder over-produces; trim to input length
        return z, y

    for step in range(1, a.steps + 1):
        x = crops(train, a.batch, rng).to(a.device)
        z, y = recon(x)
        loss = mr_stft(x, y) + 0.1 * F.l1_loss(y, x) + 1e-3 * z.pow(2).mean()   # last term bounds latent scale
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 5.0); opt.step()
        if step % 500 == 0: print(f"step {step} loss {float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)

    enc.eval(); dec.eval()
    with torch.no_grad():
        h = torch.from_numpy(held[: (len(held) // HOP) * HOP])[None, None].to(a.device)
        _, y = recon(h)
        res = float(mr_stft(h, y)); base = float(mr_stft(h, torch.zeros_like(h)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(enc.state_dict(), out / "codec_encoder.pt"); torch.save(dec.state_dict(), out / "codec_decoder.pt")
    rep = {"causal": a.causal, "steps": a.steps, "heldout_mrstft": res, "baseline_silence_mrstft": base,
           "improved_over_baseline": res < base, "manifest_sha256": _sha(a.manifest),
           "encoder_sha256": _sha(out / "codec_encoder.pt"), "decoder_sha256": _sha(out / "codec_decoder.pt"),
           "note": "no GAN loss: listen before trusting; causal checkpoints are NOT interchangeable with non-causal"}
    (out / "codec_report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))

if __name__ == "__main__": main()
