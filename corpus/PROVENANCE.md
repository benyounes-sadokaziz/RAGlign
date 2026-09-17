# Corpus Provenance

| Field | Value |
|---|---|
| Source | https://github.com/fastapi/fastapi |
| Path in source | `docs/en/docs/{tutorial,advanced,how-to}` |
| Commit | `50113da16fec53b66b80d75e80a89296de4fa5a5` |
| Vendored on | 2026-09-17 |
| Files | 97 markdown |
| Size | 541,001 bytes |
| License | MIT (FastAPI), see upstream LICENSE |

Vendored (committed to this repo) rather than fetched at runtime so that every
evaluation run is reproducible against byte-identical source documents. Character
offsets in the ground-truth spans are meaningless if the corpus can shift under them.

Reproduce:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git
cd fastapi && git checkout 50113da16fec53b66b80d75e80a89296de4fa5a5 && git sparse-checkout set docs/en/docs
```
