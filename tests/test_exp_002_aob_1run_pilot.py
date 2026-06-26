from pathlib import Path

from experiments.exp_002_aob_1run_pilot.run import run_aob_1run_pilot


def test_aob_pilot_writes_one_run_truth_tables(tmp_path: Path) -> None:
    output_dir = tmp_path / 'pilot'
    run_aob_1run_pilot(output_dir)

    expected = {
        'pilot_run_manifest.md',
        'our_result_by_case.csv',
        'same_budget_ledger.csv',
        'backend_semantics_diff.csv',
        'anti_leakage_audit.csv',
        'paper_reported_comparison.csv',
        'negative_control_audit.csv',
        'catastrophic_loss_audit.csv',
    }
    assert expected == {path.name for path in output_dir.iterdir()}


def test_aob_pilot_marks_oracle_and_reported_baselines_offline_only(tmp_path: Path) -> None:
    output_dir = run_aob_1run_pilot(tmp_path / 'pilot')
    comparison = (output_dir / 'paper_reported_comparison.csv').read_text(encoding='utf-8')
    manifest = (output_dir / 'pilot_run_manifest.md').read_text(encoding='utf-8')

    assert 'paper-reported evaluation-only baselines' in comparison
    assert 'must not enter runtime dispatch' in manifest
