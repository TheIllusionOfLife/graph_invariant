import pytest

from graph_invariant.config import Phase1Config


def test_phase1_config_defaults_are_valid():
    cfg = Phase1Config()
    assert cfg.num_train_graphs == 50
    assert cfg.num_val_graphs == 200
    assert cfg.num_test_graphs == 200
    assert cfg.max_generations == 20
    assert cfg.population_size == 5
    assert cfg.migration_interval == 10
    assert cfg.model_name == "gpt-oss:20b"
    assert cfg.enable_constrained_fallback is True
    assert cfg.stagnation_trigger_generations == 5
    assert cfg.constrained_recovery_generations == 3
    assert cfg.allow_late_constrained_recovery is True
    assert cfg.run_baselines is False
    assert cfg.persist_candidate_code_in_summary is False
    assert cfg.success_spearman_threshold == 0.85


def test_phase1_config_from_dict_overrides_values():
    cfg = Phase1Config.from_dict({"seed": 7, "num_train_graphs": 8, "target_name": "diameter"})
    assert cfg.seed == 7
    assert cfg.num_train_graphs == 8
    assert cfg.target_name == "diameter"


def test_phase1_config_from_dict_converts_island_temperatures_to_tuple():
    cfg = Phase1Config.from_dict({"island_temperatures": [0.1, 0.2, 0.3, 0.4]})
    assert cfg.island_temperatures == (0.1, 0.2, 0.3, 0.4)


def test_phase1_config_validates_constrained_fallback_thresholds():
    with pytest.raises(ValueError, match="stagnation_trigger_generations"):
        Phase1Config(stagnation_trigger_generations=0)
    with pytest.raises(ValueError, match="constrained_recovery_generations"):
        Phase1Config(constrained_recovery_generations=0)


def test_phase1_config_validates_success_threshold_range():
    with pytest.raises(ValueError, match="success_spearman_threshold"):
        Phase1Config(success_spearman_threshold=-0.1)
    with pytest.raises(ValueError, match="success_spearman_threshold"):
        Phase1Config(success_spearman_threshold=1.1)


def test_phase1_config_validates_score_weight_range_and_sum():
    with pytest.raises(ValueError, match="alpha"):
        Phase1Config(alpha=-0.1)
    with pytest.raises(ValueError, match="sum"):
        Phase1Config(alpha=0.0, beta=0.0, gamma=0.0)


def test_phase1_config_normalizes_non_unit_score_weights():
    with pytest.warns(UserWarning, match="normalizing weights"):
        cfg = Phase1Config(alpha=0.7, beta=0.2, gamma=0.2)
    assert cfg.alpha == pytest.approx(0.6363636, rel=1e-6)
    assert cfg.beta == pytest.approx(0.1818181, rel=1e-6)
    assert cfg.gamma == pytest.approx(0.1818181, rel=1e-6)


def test_phase1_config_validates_sandbox_and_pysr_budget_fields():
    with pytest.raises(ValueError, match="sandbox_max_workers"):
        Phase1Config(sandbox_max_workers=0)
    with pytest.raises(ValueError, match="pysr_niterations"):
        Phase1Config(pysr_niterations=0)
    with pytest.raises(ValueError, match="pysr_populations"):
        Phase1Config(pysr_populations=0)
    with pytest.raises(ValueError, match="pysr_timeout_in_seconds"):
        Phase1Config(pysr_timeout_in_seconds=0.0)
