import json, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT))
import model_status as ms

REG = {"components": [
    {"id": "t", "kind": "trainable", "required_for": ["x"], "script": "s.py", "command": "python s.py", "outputs": ["w.pt"],
     "gate_file": "g.json", "gate_key": "ok"},
    {"id": "e", "kind": "external", "required_for": ["x"], "outputs": ["ext/READY"], "fetch": "get it"},
    {"id": "c", "kind": "code_fix", "required_for": [], "fixed": False}]}

class T(unittest.TestCase):
    def run_in(self, files):
        d = Path(tempfile.mkdtemp())
        for f, txt in files.items():
            (d / f).parent.mkdir(parents=True, exist_ok=True); (d / f).write_text(txt)
        return {r["id"]: r["status"] for r in ms.evaluate(REG, d)}
    def test_empty(self):
        self.assertEqual(self.run_in({}), {"t": "SCRIPT_MISSING", "e": "MISSING_EXTERNAL", "c": "BLOCKED"})
    def test_needs_training(self): self.assertEqual(self.run_in({"s.py": ""})["t"], "NEEDS_TRAINING")
    def test_weights_unverified(self): self.assertEqual(self.run_in({"s.py": "", "w.pt": "x"})["t"], "WEIGHTS_PRESENT_UNVERIFIED")
    def test_gate_failed(self): self.assertEqual(self.run_in({"s.py": "", "w.pt": "x", "g.json": '{"ok": false}'})["t"], "FAILED_GATE")
    def test_gate_passed(self): self.assertEqual(self.run_in({"s.py": "", "w.pt": "x", "g.json": '{"ok": true}'})["t"], "WEIGHTS_PRESENT")
    def test_gate_null_is_unverified(self): self.assertEqual(self.run_in({"s.py": "", "w.pt": "x", "g.json": '{"ok": null}'})["t"], "WEIGHTS_PRESENT_UNVERIFIED")
    def test_real_registry_parses(self):
        rows = ms.evaluate(json.loads((ROOT / "MODEL_STATUS.json").read_text()), ROOT)
        self.assertTrue(all(r["status"] != "SCRIPT_MISSING" for r in rows), rows)
    def test_exit_codes(self):
        d = Path(tempfile.mkdtemp()); (d / "r.json").write_text(json.dumps(REG))
        self.assertEqual(ms.main(["--registry", str(d / "r.json"), "--root", str(d), "--target", "x"]), 1)
        self.assertEqual(ms.main(["--registry", str(d / "r.json"), "--root", str(d), "--target", "x", "--no-fail"]), 0)

if __name__ == "__main__": unittest.main()
