from graph_invariant.config import Phase1Config


def test_phase1_config_defaults_are_valid():
    cfg = Phase1Config()
    assert cfg.num_train_graphs == 50
    assert cfg.num_val_graphs == 200
    assert cfg.num_test_graphs == 200
    assert cfg.max_generations == 20
    assert cfg.population_size == 5
    assert cfg.migration_interval == 10


def test_phase1_config_from_dict_overrides_values():
    cfg = Phase1Config.from_dict({"seed": 7, "num_train_graphs": 8, "target_name": "diameter"})
    assert cfg.seed == 7
    assert cfg.num_train_graphs == 8
    assert cfg.target_name == "diameter"
