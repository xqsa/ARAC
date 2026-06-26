from pathlib import Path
import csv

from arac.backends.hcc import HccAobExecutionResult
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


def test_aob_pilot_uses_hcc_source_topology_not_synthetic_proxy(tmp_path: Path) -> None:
    output_dir = run_aob_1run_pilot(tmp_path / 'pilot')
    with (output_dir / 'our_result_by_case.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 24
    assert {row['source_level'] for row in rows} == {'hcc_source_topology'}
    assert {row['pilot_result_source'] for row in rows} == {'hcc_source_grounded_grouping_probe'}
    assert 'scaffold_synthetic_proxy' not in (output_dir / 'pilot_run_manifest.md').read_text(
        encoding='utf-8'
    )
    by_problem = {row['problem_id']: row for row in rows}
    assert by_problem['E1']['dimension_real'] == '1000'
    assert by_problem['S6']['dimension_real'] == '1190'
    assert by_problem['S6']['global_fes'] == '1056000'


def test_aob_pilot_can_overlay_offline_hcc_smoke_execution_result(tmp_path: Path) -> None:
    smoke_result = HccAobExecutionResult(
        problem_id='E1',
        seed=1,
        max_fes=2_000,
        final_error=42.5,
        fe_used=2_000,
        time_seconds=0.25,
        output_root=tmp_path / 'hcc-smoke',
        fresh_optimizer_execution=True,
        status='completed',
        result_source='hcc_subprocess_smoke_execution',
    )

    output_dir = run_aob_1run_pilot(
        tmp_path / 'pilot',
        smoke_execution_results=[smoke_result],
    )
    with (output_dir / 'our_result_by_case.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    by_problem = {row['problem_id']: row for row in rows}

    assert by_problem['E1']['pilot_result_source'] == 'hcc_subprocess_smoke_execution'
    assert by_problem['E1']['pilot_proxy_final_error'] == ''
    assert by_problem['E1']['hcc_smoke_final_error'] == '42.500000'
    assert by_problem['E1']['fresh_optimizer_execution'] == '1'
    assert by_problem['E1']['runtime_dispatch_allowed'] == '1'

    comparison_text = (output_dir / 'paper_reported_comparison.csv').read_text(encoding='utf-8')
    assert 'paper-reported evaluation-only baselines' in comparison_text
    assert ',0' in comparison_text
