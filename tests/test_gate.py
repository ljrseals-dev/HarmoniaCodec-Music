import hashlib, json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harmonia_gate import attest_trained, check_manifest_rights, GateRefused

class M:
    trained = False
    def mark_trained(self): self.trained = True

class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); os.chdir(self.d)
        open("w.pt", "wb").write(os.urandom(2_000_000)); self.h = hashlib.sha256(open("w.pt", "rb").read()).hexdigest()
        open("a.wav", "wb").write(b"x")
        self.man = {"checkpoint_sha256": self.h, "recordings": [{"path": "a.wav", "rights": "own", "consent_doc_sha256": "a" * 64, "speaker_id": "s"}]}
        self.ev = {"checkpoint_sha256": self.h, "heldout_latent_l1": 0.2, "baseline_zero_latent_l1": 0.8, "canary_pass": True}
    def run_gate(self, man=None, ev=None):
        json.dump(man or self.man, open("m.json", "w")); json.dump(ev or self.ev, open("e.json", "w")); m = M()
        try: attest_trained(m, "w.pt", "m.json", "e.json")
        except GateRefused: pass
        return m.trained
    def test_good(self): self.assertTrue(self.run_gate())
    def test_scraped(self):
        bad = json.loads(json.dumps(self.man)); bad["recordings"][0]["rights"] = "scraped"; self.assertFalse(self.run_gate(man=bad))
    def test_worse_than_baseline(self): self.assertFalse(self.run_gate(ev={**self.ev, "heldout_latent_l1": 0.9}))
    def test_canary_none_refused(self): self.assertFalse(self.run_gate(ev={**self.ev, "canary_pass": None}))
    def test_stale_binding(self): self.assertFalse(self.run_gate(man={**self.man, "checkpoint_sha256": "0" * 64}))
    def test_manifest_rights_check(self):
        json.dump(self.man, open("m.json", "w")); check_manifest_rights("m.json")
        json.dump({"recordings": []}, open("m.json", "w"))
        with self.assertRaises(GateRefused): check_manifest_rights("m.json")

if __name__ == "__main__": unittest.main()
