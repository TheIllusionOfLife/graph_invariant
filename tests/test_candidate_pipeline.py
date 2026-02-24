"""Tests for candidate_pipeline module: prompt building, repair, and error helpers."""

from graph_invariant.candidate_pipeline import (
    _candidate_prompt,
    _record_recent_failure,
    _summarize_error_details,
    _topology_descriptor,
    _update_prompt_mode_after_generation,
)
from graph_invariant.config import Phase1Config
from graph_invariant.types import CheckpointState


def test_constrained_mode_allows_late_recovery_by_default():
    cfg = Phase1Config(
        constrained_recovery_generations=2,
        run_baselines=False,
    )
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: []},
        island_prompt_mode={0: "constrained"},
        island_constrained_generations={0: 3},
        island_stagnation={0: 3},
    )

    _update_prompt_mode_after_generation(cfg, state, island_id=0, had_valid_train_candidate=True)
    assert state.island_prompt_mode[0] == "free"


def test_constrained_mode_can_forbid_late_recovery():
    cfg = Phase1Config(
        constrained_recovery_generations=2,
        allow_late_constrained_recovery=False,
        run_baselines=False,
    )
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: []},
        island_prompt_mode={0: "constrained"},
        island_constrained_generations={0: 3},
        island_stagnation={0: 3},
    )

    _update_prompt_mode_after_generation(cfg, state, island_id=0, had_valid_train_candidate=True)
    assert state.island_prompt_mode[0] == "constrained"


def test_summarize_error_details_uses_detail_for_top_category():
    details = [
        {
            "value": None,
            "error_type": "timeout",
            "error_detail": "candidate evaluation timed out",
        },
        {
            "value": None,
            "error_type": "runtime_exception",
            "error_detail": "ZeroDivisionError: division by zero",
        },
        {
            "value": None,
            "error_type": "runtime_exception",
            "error_detail": "TypeError: bad operand",
        },
    ]
    summary = _summarize_error_details(details)
    assert summary.startswith("runtime_exception:")
    assert "timed out" not in summary


def test_record_recent_failure_deduplicates_and_keeps_recency():
    state = CheckpointState(experiment_id="exp", generation=0, islands={0: []})
    _record_recent_failure(state, island_id=0, failure_text="a", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="b", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="a", max_items=3)
    _record_recent_failure(state, island_id=0, failure_text="c", max_items=3)
    assert state.island_recent_failures[0] == ["b", "a", "c"]


def test_topology_descriptor_handles_missing_feature_keys():
    descriptor = _topology_descriptor(
        y_pred_valid=[1.0, 2.0, 3.0],
        features_val=[{"density": 0.2}, {"density": 0.3}, {"density": 0.4}],
        valid_indices=(0, 1, 2),
    )
    assert 0.0 <= descriptor[0] <= 1.0
    assert 0.0 <= descriptor[1] <= 1.0


def test_candidate_prompt_maps_island_0_to_refinement():
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=0, target_name="diameter")
    assert any(word in prompt.lower() for word in ("improve", "refine"))


def test_candidate_prompt_maps_island_1_to_combination():
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=1, target_name="diameter")
    assert "combine" in prompt.lower()


def test_candidate_prompt_maps_island_3_to_novel():
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    prompt = _candidate_prompt(state, island_id=3, target_name="diameter")
    assert any(word in prompt.lower() for word in ("new", "novel"))


def test_candidate_prompt_passes_fitness_mode(monkeypatch):
    """_candidate_prompt should forward fitness_mode to build_prompt."""
    import inspect

    from graph_invariant.llm_ollama import build_prompt as original_build

    captured: dict[str, str] = {}
    sig = inspect.signature(original_build)

    def spy_build(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        captured["fitness_mode"] = bound.arguments["fitness_mode"]
        return original_build(*args, **kwargs)

    monkeypatch.setattr("graph_invariant.candidate_pipeline.build_prompt", spy_build)

    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    _candidate_prompt(state, island_id=0, target_name="diameter", fitness_mode="upper_bound")
    assert captured["fitness_mode"] == "upper_bound"


def test_candidate_prompt_includes_archive_exemplars():
    state = CheckpointState(
        experiment_id="exp",
        generation=0,
        islands={0: [], 1: [], 2: [], 3: []},
    )
    exemplar_codes = ["def new_invariant(s): return s['n'] + 1"]
    prompt = _candidate_prompt(
        state,
        island_id=1,
        target_name="diameter",
        archive_exemplars=exemplar_codes,
    )
    assert "s['n'] + 1" in prompt
