# Training-data manifest

`manifest.json` lists every recording used for training. Training scripts refuse to start unless every entry passes. See [`manifest.example.json`](../manifest.example.json).

| Field | Required | Meaning |
|---|---|---|
| `path` | yes | 24 kHz mono 16-bit PCM wav (must exist) |
| `features` | for `train_synth.py` | `.npz` with `phon`, `f0`, `voiced`, `loud` @120 frames/s |
| `rights` | yes | `own` (your voice) or `written` (written permission from the singer) |
| `consent_doc_sha256` | yes | SHA-256 of the signed consent / ownership document you keep on file |
| `speaker_id` | yes | stable identifier for the singer |

The last recording in the list is held out for evaluation. Do not include scraped, commercial or third-party material without written rights.
