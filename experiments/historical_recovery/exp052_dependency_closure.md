# EXP-052 dependency closure audit

- First-level local imports checked: `30`
- Status counts: `{'exact_session_boundary_source': 27, 'exact_session_source': 1, 'exact_git_worktree_source': 2}`
- Exact first-level closure: **yes**
- Replay authorized: **no**

| Path | Status | SHA-256 |
|---|---|---|
| `src/arac/actions/controller_profiles.py` | **exact_session_boundary_source** | `837cf684b9bdd9375da8aead1e67d7682f2c1a9a8d6a94c39d25859776b74090` |
| `src/arac/actions/ctp.py` | **exact_session_boundary_source** | `0b71a603fb4eb065099146dfb335f5927ab1f1b5171018bf58a5ce99eb9d79ee` |
| `src/arac/actions/ctp_stable.py` | **exact_session_boundary_source** | `d78084481cac9ff1a5ed4d6899eb677efa5bada500dac035b1846f27305499b6` |
| `src/arac/actions/gcb.py` | **exact_session_boundary_source** | `260e140547d6aa9d04b24435db4a8494646bda0fe7c9b4ab15c772eb39593156` |
| `src/arac/actions/gcb_ensemble.py` | **exact_session_boundary_source** | `78349dab923cfd8e1dded975de14dd19b5278b46d47d7659f37044d0313f90c2` |
| `src/arac/actions/group_optimizer_type.py` | **exact_session_boundary_source** | `14bfb2cc1a5e684eff3da6fd9f1cef7c1aadb74f4b4cba1b2cbc5238f9043cb1` |
| `src/arac/actions/runtime_dispatcher.py` | **exact_session_boundary_source** | `75de1a3dcd955f62bd615d6ced48cf0769c16c4cb4890b8c9aa8c9eaac74a02f` |
| `src/arac/actions/shared_variable_blend.py` | **exact_session_boundary_source** | `5b3c4a6dc214a66a1745707be6d2ce78860def1b5366f933771b93fbeaa31e1c` |
| `src/arac/actions/smp.py` | **exact_session_source** | `5130b18a39c8f76e0713a30026fd9a997d34e982c7d7f8287b6d1047161efc16` |
| `src/arac/backends/hcc.py` | **exact_session_boundary_source** | `f1a15f33c4d9e62f26805915e7af42808a9b48838983c2cd7e836169fc6e10d8` |
| `src/arac/backends/hcc_action_ceiling.py` | **exact_session_boundary_source** | `c63154ec5a1cfe7a003b4767d293fde1f40d2c0185139eaaebe2a1ad977bd0ab` |
| `src/arac/backends/hcc_action_ceiling_runtime.py` | **exact_session_boundary_source** | `a5208c21dfdba2164930fce88d997d209a5cfeb9bb2559fe5b594bd118026ee1` |
| `src/arac/backends/hcc_evidence_overlay.py` | **exact_session_boundary_source** | `d7ac2b3a3b2e6767044fe3ded45dae2dc1c531715ff43279032ca5735342dbe6` |
| `src/arac/backends/hcc_gcb.py` | **exact_session_boundary_source** | `1ec29d92af7698da96c9cc380dbf6f4b77276a3aef3669fcc591898e86370ead` |
| `src/arac/backends/hcc_gcb_ensemble.py` | **exact_session_boundary_source** | `0e1d1da9bd967d93b81347505858cccde82fb43e5f7b060670f7bac9ffe93c82` |
| `src/arac/backends/hcc_phase2_action_context.py` | **exact_session_boundary_source** | `67f6bead5626edfcc91661854052e4e078d4449aa7fef47f010a616f8ceea45c` |
| `src/arac/evidence/overlap_relation_builder.py` | **exact_session_boundary_source** | `d7f91b1fc0b4e7b4afcf9a5490dc6074aa0eae7d7aec69306a10fa2fc105a968` |
| `src/arac/evidence/trajectory_accumulator.py` | **exact_session_boundary_source** | `ec5480af644cdf985540710d76189911221b03c5db476958c1c5b641918b1d65` |
| `src/arac/policy/action_ceiling.py` | **exact_session_boundary_source** | `6375829fdc67a5cbc491874f04d5d8b3a03699511bc7601d4fa3a4ec0aea93a3` |
| `src/arac/policy/action_trust_policy.py` | **exact_session_boundary_source** | `8ece5dec475918cb0d114891bfa746288f76c91737b5e5355c7d0f6dfdf40864` |
| `src/arac/policy/evidence_model.py` | **exact_session_boundary_source** | `a3d2fb1d344d96cf34b3e5b8623f76b91bb186cde4126d325aebdf2b668fd7ad` |
| `src/arac/policy/evidence_overlay.py` | **exact_session_boundary_source** | `c0ce4ed81b7582f148af69812eaf22acacb91430a824efab6ba096302dee8063` |
| `src/arac/policy/relation_policy.py` | **exact_session_boundary_source** | `bf0a75d7894c812d8ed57c4b64118e082c92d05cb3b6ec2026aaa76294f1345e` |
| `vendor/hcc/AOB/AOB.py` | **exact_git_worktree_source** | `538397a58bf44ccfe2b9159a81687c7ffcac8b3d6f811f43e93ef0d17a55524c` |
| `vendor/hcc/AOB/utils.py` | **exact_git_worktree_source** | `aaf775be4ddb0dfc6a7696014e81472ec246dc4d8745e2909f13959fcf716cbc` |
| `vendor/hcc/HCC/NDAs/MMES/mmes.py` | **exact_session_boundary_source** | `25487615d60bb0dfc85a968a64f46ce5b14e7759baed4adbfe34a3648a1dafdd` |
| `vendor/hcc/HCC/NDAs/MMES/state.py` | **exact_session_boundary_source** | `b258f7060f29f2acc2be1ae43cb1d7146b36cc7009b630f5d187ae1be5f6cba2` |
| `vendor/hcc/HCC/OPT/CMAES/cmaes.py` | **exact_session_boundary_source** | `128a208adfee4f7ec548b5db6415280fa323693726bab7caca9f22be9447e5c1` |
| `vendor/hcc/HCC/OPT/CMAES/sepcmaes.py` | **exact_session_boundary_source** | `43c104d8560a01ffdad4d073e808a6bd557ef2c4dfae20664aba996db580ecd3` |
| `vendor/hcc/HCC/RDDSM.py` | **exact_session_boundary_source** | `e8a683d0255bdd00a7c131ad2efba4cb89c9b1de2e2644c09beb3405edacd9bc` |

The recovered runner, CTP/GCB/SMP action modules, and HCC optimizer sources
are exact at the formal-start boundary. The two AOB Python modules are
exact Git-worktree sources: their last Git changes predate EXP-052, the
retained initial status is complete and clean for both paths, and neither
path has a pre-run session patch. They are still not receipt-hash-bound.

The separate environment audit in `exp052_environment.md` finds matching
runtime pins (`numpy`, `scipy`, `PyYAML`) in the current project `.venv`,
but the receipt records no Python/dependency/environment-manifest hash.
This closes the source description, not the replay gate.
