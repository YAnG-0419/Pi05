"""The three profiles share devices while keeping model and planner choices independent."""
import json
import sys

import numpy as np
import pytest

from deployment import hardware, slow
from deployment.models import joint54, joint64
from deployment.normal import deploy as normal
from deployment.profiles import PROFILES, get_profile
from deployment import server
from experiments.weight_motion_eval import planner
from fr3_wuji.paths import ROOT

CONFIG = ROOT / 'experiments/weight_motion_eval/config.yaml'


def checkpoint(tmp_path, dim):
    root = tmp_path / str(dim)
    (root / 'params').mkdir(parents=True)
    (root / 'params/manifest.ocdbt').write_bytes(b'test-only-not-a-model')
    (root / 'assets/test').mkdir(parents=True)
    stats = {key: {'mean': [0.] * dim, 'std': [1.] * 54 + [0.] * (dim-54),
                   'q01': [-1.] * 54 + [0.] * (dim-54), 'q99': [1.] * 54 + [0.] * (dim-54)}
             for key in ('state', 'actions')}
    (root/'assets/test/norm_stats.json').write_text(json.dumps({'norm_stats': stats}))
    return root


def test_profiles_choose_engines_and_adapters_explicitly():
    assert {(p.name, p.engine, p.adapter) for p in PROFILES.values()} == {
        ('slow54', 'slow', 'joint54'), ('slow64', 'slow', 'joint64'), ('normal', 'normal', 'joint54'),
    }
    assert normal.hardware is slow.hardware is hardware
    assert get_profile('normal').checkpoint == get_profile('slow54').checkpoint
    assert get_profile('normal').uri == get_profile('slow54').uri
    with pytest.raises(ValueError):
        get_profile('normal64')


def test_slow_profiles_do_not_mutate_planner_or_each_other():
    validator, reader = planner.validate_config, planner.read_config
    default = reader(CONFIG)
    adapted = reader(CONFIG, profile='slow64')
    assert adapted['arm_raw_initial_delta_rad'] == 1.5
    assert adapted['max_smoothing_delta_rad']['arm'] == 0.2
    assert reader(CONFIG) == default
    assert default['arm_raw_initial_delta_rad'] == 0.3
    assert default['max_smoothing_delta_rad']['arm'] == 0.03
    assert (planner.validate_config, planner.read_config) == (validator, reader)
    with pytest.raises(ValueError):
        validator(dict(default, arm_raw_initial_delta_rad=1.5))
    with pytest.raises(ValueError):
        validator(dict(adapted, arm_raw_initial_delta_rad=1.500001))
    with pytest.raises(ValueError):
        reader(CONFIG, profile='unknown')


def test_explicit_planning_profile_survives_plan_construction():
    raw = np.zeros((50, 54))
    raw[:, 0] = 0.4
    limits = {'lower': np.full(54, -2.), 'upper': np.full(54, 2.), 'speed': np.full(54, 2.)}
    with pytest.raises(ValueError):
        planner.build_plan(raw, np.zeros(54), planner.read_config(CONFIG), limits, [str(i) for i in range(54)])
    plan = planner.build_plan(raw, np.zeros(54), planner.read_config(CONFIG, profile='slow64'), limits, [str(i) for i in range(54)])
    assert plan.config['planning_profile'] == 'slow64'
    assert plan.config['arm_raw_initial_delta_rad'] == 1.5


def test_slow_cli_selects_defaults_without_changing_standard_profile():
    a = slow.parse_args([], profile=get_profile('slow64'))
    b = slow.parse_args([], profile=get_profile('slow54'))
    assert a.uri.endswith(':8002') and b.uri.endswith(':8001')
    assert b.checkpoint == ROOT/'checkpoints/pi05/19999'
    with pytest.raises(ValueError):
        slow.parse_args([], profile=get_profile('normal'))


def test_adapters_reject_cross_connected_dimensions(tmp_path):
    small, big = checkpoint(tmp_path, 54), checkpoint(tmp_path, 64)
    assert joint54.checkpoint_contract(small)[0]['action_dim'] == 54
    assert joint64.checkpoint_contract(big)[0]['model_action_dim'] == 64
    with pytest.raises(ValueError, match='54 finite'):
        joint54.checkpoint_contract(big)
    with pytest.raises(ValueError, match='64 finite'):
        joint64.checkpoint_contract(small)


@pytest.mark.parametrize('profile,dim', [('slow54', 54), ('slow64', 64), ('normal', 54)])
def test_server_contract_check_never_loads_model(profile, dim, tmp_path, monkeypatch, capsys):
    path = checkpoint(tmp_path, dim)
    monkeypatch.setitem(sys.modules, 'jax', None)
    server.main(['--profile', profile, '--checkpoint', str(path), '--check-only'])
    report = json.loads(capsys.readouterr().out)
    assert report['profile'] == profile
    assert report['hardware_output'] is False
    assert report['action_dim'] == 54


def test_runtime_contract_follows_selected_adapter(tmp_path, monkeypatch):
    # Only serialize a launch configuration; no process or device calls.
    monkeypatch.setattr(hardware, 'verify_overlay', lambda: {'built_library': 'test-only', 'built_library_sha256': 'test'})
    for name, dim in [('slow54', 54), ('slow64', 64)]:
        profile = get_profile(name)
        path = checkpoint(tmp_path, dim)
        args = slow.parse_args(['--checkpoint', str(path)], profile=profile)
        output = tmp_path/name
        output.mkdir()
        runtime, _, _ = hardware.write_runtime(args, output, profile=profile)
        assert runtime['deployment_profile'] == name
        assert runtime['model_adapter'] == profile.adapter
        assert runtime['checkpoint']['action_dim'] == 54
        assert ('model_action_dim' in runtime['checkpoint']) == (dim == 64)


@pytest.mark.parametrize('strategy', ['slow', 'normal'])
@pytest.mark.parametrize('model', ['19999', '30000'])
def test_strategy_and_pi05_model_are_independent(strategy, model):
    from deployment.profiles import compose_profile
    profile = compose_profile(strategy, model)
    engine = slow if strategy == 'slow' else normal
    args = engine.parse_args([], profile=profile)
    assert profile.engine == strategy
    assert profile.adapter == ('joint54' if model == '19999' else 'joint64')
    assert args.checkpoint == ROOT/'checkpoints/pi05'/model
    assert args.uri.endswith(':8001' if model == '19999' else ':8002')
    # Switching weights does not implicitly change execution/acquisition limits.
    assert profile.planning_profile == 'slow54'


def test_extended_limits_are_explicit_and_legacy_defaults_stay_available():
    from deployment.profiles import compose_profile
    assert compose_profile('slow', '30000').planning_profile == 'slow54'
    assert compose_profile('slow', '30000', planning_profile='slow64').planning_profile == 'slow64'
    assert get_profile('slow64').planning_profile == 'slow64'
    with pytest.raises(ValueError, match='standard'):
        compose_profile('normal', '30000', planning_profile='slow64')


@pytest.mark.parametrize('model,dim', [('19999', 54), ('30000', 64)])
def test_model_service_selection_does_not_require_an_execution_strategy(model, dim, tmp_path, monkeypatch, capsys):
    path = checkpoint(tmp_path, dim)
    monkeypatch.setitem(sys.modules, 'jax', None)
    server.main(['--model', model, '--checkpoint', str(path), '--check-only'])
    report = json.loads(capsys.readouterr().out)
    assert report['adapter'] == ('joint54' if dim == 54 else 'joint64')
    assert report['action_dim'] == 54


def test_normal_64_serializes_the_64d_model_contract(tmp_path, monkeypatch):
    from deployment.profiles import compose_profile
    profile = compose_profile('normal', '30000')
    monkeypatch.setattr(hardware, 'verify_overlay', lambda: {'built_library': 'test-only', 'built_library_sha256': 'test'})
    path = checkpoint(tmp_path, 64)
    args = normal.parse_args(['--checkpoint', str(path)], profile=profile)
    out = tmp_path/'normal64'
    out.mkdir()
    runtime, _, _ = hardware.write_runtime(args, out, profile=profile)
    assert runtime['execution_engine'] == 'normal'
    assert runtime['checkpoint']['model_action_dim'] == 64
    assert runtime['checkpoint']['action_dim'] == 54
    assert runtime['checkpoint']['action_representation'] == 'absolute_joint_positions'


@pytest.mark.parametrize('strategy', ['slow', 'normal'])
@pytest.mark.parametrize('model', ['19999', '30000'])
def test_run_dispatch_passes_both_selections_without_device_access(strategy, model, monkeypatch):
    from deployment import run
    calls = []
    target = slow if strategy == 'slow' else normal
    monkeypatch.setattr(target, 'main', lambda argv, *, profile: calls.append((argv, profile)))
    run.main(['--strategy', strategy, '--model', model, '--check'])
    assert len(calls) == 1
    assert calls[0][0] == ['--check']
    assert calls[0][1].engine == strategy
    assert calls[0][1].checkpoint == ROOT/'checkpoints/pi05'/model
