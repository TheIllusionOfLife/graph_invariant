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
    assert cfg.run_baselines is True
    assert cfg.persist_candidate_code_in_summary is False
    assert cfg.success_spearman_threshold == 0.85
    assert cfg.benchmark_seeds == (11, 22, 33, 44, 55)
    assert cfg.novelty_bootstrap_samples == 1000
    assert cfg.enforce_pysr_parity_for_success is True
    assert cfg.require_baselines_for_success is True
    assert cfg.persist_prompt_and_response_logs is False
    assert cfg.novelty_threshold == 0.7
    assert cfg.pysr_parity_epsilon == 0.0
    assert cfg.llm_timeout_sec == 60.0
    assert cfg.enable_self_correction is True
    assert cfg.self_correction_max_retries == 1
    assert cfg.self_correction_feedback_window == 3
    assert cfg.enable_spectral_feature_pack is True
    assert cfg.ood_train_special_topology_ratio == 0.1
    assert cfg.ood_val_special_topology_ratio == 0.1
    assert cfg.enable_dual_map_elites is False
    assert cfg.map_elites_topology_bins == 5


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
    with pytest.raises(ValueError, match="pysr_procs"):
        Phase1Config(pysr_procs=-1)
    with pytest.raises(ValueError, match="pysr_timeout_in_seconds"):
        Phase1Config(pysr_timeout_in_seconds=0.0)


def test_phase1_config_validates_benchmark_and_novelty_settings():
    with pytest.raises(ValueError, match="benchmark_seeds"):
        Phase1Config(benchmark_seeds=())
    with pytest.raises(ValueError, match="novelty_bootstrap_samples"):
        Phase1Config(novelty_bootstrap_samples=0)
    with pytest.raises(ValueError, match="novelty_threshold"):
        Phase1Config(novelty_threshold=-0.1)
    with pytest.raises(ValueError, match="novelty_threshold"):
        Phase1Config(novelty_threshold=1.1)
    with pytest.raises(ValueError, match="pysr_parity_epsilon"):
        Phase1Config(pysr_parity_epsilon=-1e-3)
    with pytest.raises(ValueError, match="llm_timeout_sec"):
        Phase1Config(llm_timeout_sec=0.0)


def test_novelty_gate_threshold_default():
    cfg = Phase1Config()
    assert cfg.novelty_gate_threshold == 0.15


def test_novelty_gate_threshold_validates_range():
    with pytest.raises(ValueError, match="novelty_gate_threshold"):
        Phase1Config(novelty_gate_threshold=-0.1)
    with pytest.raises(ValueError, match="novelty_gate_threshold"):
        Phase1Config(novelty_gate_threshold=1.1)


def test_phase1_config_validates_self_correction_fields():
    with pytest.raises(ValueError, match="self_correction_max_retries"):
        Phase1Config(self_correction_max_retries=-1)
    with pytest.raises(ValueError, match="self_correction_feedback_window"):
        Phase1Config(self_correction_feedback_window=0)


# ── Bounds mode config tests ────────────────────────────────────────


def test_fitness_mode_defaults_to_correlation():
    cfg = Phase1Config()
    assert cfg.fitness_mode == "correlation"


def test_fitness_mode_accepts_valid_modes():
    for mode in ("correlation", "upper_bound", "lower_bound"):
        cfg = Phase1Config(fitness_mode=mode)
        assert cfg.fitness_mode == mode


def test_fitness_mode_rejects_invalid_mode():
    with pytest.raises(ValueError, match="fitness_mode"):
        Phase1Config(fitness_mode="invalid")


def test_bound_tolerance_defaults_and_validates():
    cfg = Phase1Config()
    assert cfg.bound_tolerance == 1e-9
    with pytest.raises(ValueError, match="bound_tolerance"):
        Phase1Config(bound_tolerance=-1.0)


def test_bound_score_thresholds_default_and_validate():
    cfg = Phase1Config()
    assert cfg.success_bound_score_threshold == 0.7
    assert cfg.success_satisfaction_threshold == 0.95
    with pytest.raises(ValueError, match="success_bound_score_threshold"):
        Phase1Config(success_bound_score_threshold=-0.1)
    with pytest.raises(ValueError, match="success_satisfaction_threshold"):
        Phase1Config(success_satisfaction_threshold=1.5)


def test_run_baselines_defaults_to_true():
    cfg = Phase1Config()
    assert cfg.run_baselines is True


# ── MAP-Elites config tests ─────────────────────────────────────────


def test_enable_map_elites_defaults_to_false():
    cfg = Phase1Config()
    assert cfg.enable_map_elites is False


def test_map_elites_bins_defaults_to_5():
    cfg = Phase1Config()
    assert cfg.map_elites_bins == 5


def test_map_elites_bins_validates_minimum():
    with pytest.raises(ValueError, match="map_elites_bins"):
        Phase1Config(map_elites_bins=1)


def test_map_elites_bins_accepts_valid_value():
    cfg = Phase1Config(map_elites_bins=10)
    assert cfg.map_elites_bins == 10


def test_topology_ood_ratios_validate_range():
    with pytest.raises(ValueError, match="ood_train_special_topology_ratio"):
        Phase1Config(ood_train_special_topology_ratio=-0.1)
    with pytest.raises(ValueError, match="ood_train_special_topology_ratio"):
        Phase1Config(ood_train_special_topology_ratio=1.1)
    with pytest.raises(ValueError, match="ood_val_special_topology_ratio"):
        Phase1Config(ood_val_special_topology_ratio=-0.1)
    with pytest.raises(ValueError, match="ood_val_special_topology_ratio"):
        Phase1Config(ood_val_special_topology_ratio=1.1)


def test_warns_when_dual_map_elites_enabled_without_map_elites():
    with pytest.warns(UserWarning, match="enable_dual_map_elites"):
        Phase1Config(enable_map_elites=False, enable_dual_map_elites=True)


def test_map_elites_topology_bins_validates_minimum():
    with pytest.raises(ValueError, match="map_elites_topology_bins"):
        Phase1Config(map_elites_topology_bins=1)


# ── Flexible island count tests ──────────────────────────────────────


def test_island_temperatures_accepts_two_islands():
    cfg = Phase1Config(island_temperatures=(0.5, 0.8))
    assert cfg.island_temperatures == (0.5, 0.8)


def test_island_temperatures_accepts_three_islands():
    cfg = Phase1Config(island_temperatures=(0.3, 0.5, 1.0))
    assert cfg.island_temperatures == (0.3, 0.5, 1.0)


def test_island_temperatures_accepts_six_islands():
    cfg = Phase1Config(island_temperatures=(0.2, 0.3, 0.5, 0.7, 0.9, 1.2))
    assert len(cfg.island_temperatures) == 6


def test_island_temperatures_rejects_empty():
    with pytest.raises(ValueError, match="island_temperatures"):
        Phase1Config(island_temperatures=())


def test_island_temperatures_accepts_single_island():
    cfg = Phase1Config(island_temperatures=(0.5,))
    assert cfg.island_temperatures == (0.5,)
