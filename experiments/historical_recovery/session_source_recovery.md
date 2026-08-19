# Session source recovery

The retained Codex session contains enough patch evidence to recover the
execution-critical historical runner bytes without restoring them into the
production runtime.

- Session: `C:\Users\83718\.codex\sessions\2026\07\26\rollout-2026-07-26T17-04-21-019f9dab-0f42-70f2-8f45-5cf51411e668.jsonl`
- Patch events parsed: `965`
- Retained campaign files: `21`
- Performance experiments started: **no**

| Lane | Exact execution source | SHA-256 | Recovery method |
|---|---|---|---|
| AOR | **yes** | `2d870d14fa536dee488d45a69abea19e50e86dc20748b026d5fc4a16afcb4165` | full session content |
| CTP | **yes** | `9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594` | reverse patch chain |
| GCB | **yes** | `9fe50c891534a80c2d95c8f340f5a5e6dabdf8429e46fdc74d6d38389845d594` | reverse patch chain |
| SMP | **yes** | `b17021e8ffe1de76fea48b52ed3c00a62b4cc93bf4c2c759604064d14ebc68ac` | reverse patch chain |

## Gate status

All four execution-source hashes are recovered exactly. EXP-052 starts
from a fresh seeded run rather than an external checkpoint. Replay remains
blocked because the historical dependency closure, optimizer lifecycle,
and numerical environment have not yet been proven complete.
