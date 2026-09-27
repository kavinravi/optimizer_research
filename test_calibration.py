"""Calibration selection/retention checks; run with python test_calibration.py."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from artifacts import atomic_json, fingerprint
from calibration import (boundary_arms, calibrate, candidates, finish_report,
                         make_calibration_plan, prune_checkpoints, ranked)
from optimizers import optimizer_for
from study import check_plan
from test_training import fixture, assert_nested
from prepare_data import prepare
from train import Config, model_for, read_checkpoint, source_identity, train
import unittest


def main():
    spec = json.loads(Path('calibration.json').read_text())
    baseline = candidates(spec, ['adamw'])
    assert len(baseline) == 48
    selected = {}
    for c in baseline:
        selected.setdefault(f"{c['architecture']}/{c['size']}/adamw", [dict(config=c)])
    others = candidates(spec, ['muon', 'shampoo', 'kfac'], selected)
    assert len(others) == 144
    assert all(c['fallback_lr'] == selected[f"{c['architecture']}/{c['size']}/adamw"][0]['config']['lr'] for c in others)
    for c in baseline + others:
        Config(**c).validate()
    # The optimizer library must actually receive every newly exposed setting.
    tiny = model_for('transformer', '150m', 8, True, vocab_size=9)
    opt, _ = optimizer_for(tiny, 'shampoo', lr=.003, fallback_lr=.001,
                           momentum=.9, shampoo_graft='adagrad', fallback_beta2=.99,
                           fallback_weight_decay=.01, refresh=50)
    assert all(g['momentum'] == .9 and g['graft'] == 'adagrad' and g['precondition_frequency'] == 50 for g in opt.param_groups)
    assert all(g['adamw_betas'] == (.9, .99) for g in opt.param_groups)
    assert all(g['weight_decay'] in (0, .01) for g in opt.param_groups if not g['use_preconditioner'])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tokenizer, recipe = fixture(root)
        data = root / 'data'
        prepare(data, tokenizer, recipe)
        sources = source_identity()
        configs = [asdict(Config(device='cpu', precision='fp32', sequence=8, accumulation=2,
                                eval_tokens=64, lr=lr, fallback_lr=lr)) for lr in (.001, .003)]
        plan = make_calibration_plan(spec, 'check', configs, [1337, 7331], 96, data, sources)
        results = root / 'runs'
        results.mkdir()
        for trial in plan['trials']:
            d = results / trial['id']
            d.mkdir()
            c = trial['config']
            atomic_json(d / 'config.json', c)
            atomic_json(d / 'status.json', dict(status='complete', tokens=96, train_seconds=2, wall_seconds=3))
            atomic_json(d / 'metadata.json', dict(identity=dict(config_id=fingerprint(c), data_id=plan['data_id'], sources=sources), hardware=dict(device='cpu')))
            # The isolated best seed belongs to the worse mean candidate.
            loss = (1 if c['seed'] == 1337 else 5) if c['lr'] == .001 else 2.5
            (d / 'metrics.jsonl').write_text(json.dumps(dict(kind='val', tokens=48, loss=loss+.1)) + '\n' + json.dumps(dict(kind='val', tokens=96, loss=loss)) + '\n')
            (d / 'step_00000006.pt').write_bytes(b'fixture')
        ranking, _ = ranked([plan], results)
        winner = ranking['transformer/150m/adamw'][0]
        assert winner['config']['lr'] == .003 and winner['mean_loss'] == 2.5
        assert boundary_arms(ranking, configs) == ['transformer/150m/adamw']
        prune_checkpoints(plan, results, {winner['candidate']})
        assert sum(p.exists() for p in results.glob('*/step_*.pt')) == 2
        assert ranked([plan], results)[0] == ranking
        broken = results / plan['trials'][0]['id'] / 'status.json'
        before = json.loads(broken.read_text())
        atomic_json(broken, dict(status='failed', error='PermissionError: missing cache directory'))
        try:
            ranked([plan], results)
            raise AssertionError('Infrastructure failure was treated as an optimizer result')
        except RuntimeError as exc:
            assert 'Infrastructure' in str(exc)
        atomic_json(broken, before)
        # Readiness must never launch training, and an unresolved boundary must
        # prevent generation of an executable final plan.
        short_spec = spec | dict(architectures=['transformer'], sizes=['150m'], horizon_tokens=96)
        horizon = dict(plan, trials=[t for t in plan['trials'] if t['candidate'] == winner['candidate']][:1])
        manifest = json.loads((data/'manifest.json').read_text())
        manifest['splits']['train']['tokens'] = 536870913
        positive, negative = root/'ready', root/'not-ready'
        positive.mkdir()
        negative.mkdir()
        with patch('calibration.verify_data', return_value=manifest):
            finish_report(short_spec, positive, ranking, horizon, results, [])
            finish_report(short_spec, negative, ranking, horizon, results, ['LR boundary'])
        proposed = json.loads((positive/'final-plan.json').read_text())
        check_plan(proposed)
        assert {t['config']['seed'] for t in proposed['trials']} == {101, 202, 303}
        assert all(t['config']['phase'] == 'final' and t['config']['evaluate_test'] for t in proposed['trials'])
        assert not (negative/'final-plan.json').exists()
        assert json.loads((positive/'main-study-readiness.json').read_text())['main_ready']
        assert not json.loads((negative/'main-study-readiness.json').read_text())['main_ready']
        # Exercise the entire staged controller with recorded trainer outputs.
        # Actual training/resume is checked below; no GPU is needed for orchestration.
        calls = []
        def recorded_run(plan_path, outputs, gpus, **kwargs):
            p = json.loads(Path(plan_path).read_text())
            calls.append(p['stage'])
            for trial in p['trials']:
                cfg = trial['config']
                assert not cfg['evaluate_test'] and cfg['phase'] == 'tune'
                out = Path(outputs)/trial['id']
                out.mkdir(parents=True, exist_ok=True)
                if (out/'status.json').exists():
                    continue
                atomic_json(out/'config.json', cfg)
                atomic_json(out/'status.json', dict(status='complete', tokens=cfg['total_tokens'], train_seconds=2, wall_seconds=3))
                atomic_json(out/'metadata.json', dict(identity=dict(config_id=fingerprint(cfg), data_id=p['data_id'], sources=p['sources']), hardware=dict(device='cpu')))
                target = dict(adamw=.001, muon=.02, shampoo=.01, kfac=.003)[cfg['optimizer']]
                loss = 3 + abs(cfg['lr']-target)*100
                (out/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(kind='val', tokens=t, loss=l)) for t, l in
                    [(cfg['total_tokens']//2, loss+.1), (cfg['total_tokens'], loss)])+'\n')
                (out/'step_00000001.pt').write_bytes(b'fixture')
            return True
        controller_spec = short_spec | dict(data=str(data), screen_tokens=96, confirm_tokens=192,
                                             training=spec['training'] | dict(sequence=8, accumulation=2, eval_tokens=64))
        spec_path = root/'controller.json'
        atomic_json(spec_path, controller_spec)
        campaign_root = root/'campaign'
        with patch('calibration.verify_data', return_value=manifest), patch('calibration.run_plan', side_effect=recorded_run):
            calibrate(spec_path, campaign_root, ['cpu'])
            assert calls == ['adamw-screen', 'adamw-expand', 'adamw-confirm', 'others-screen', 'others-expand', 'others-confirm', 'horizon']
            assert json.loads((campaign_root/'main-study-readiness.json').read_text())['main_ready']
            frozen = (campaign_root/'final-plan.json').read_text()
            calibrate(spec_path, campaign_root, ['cpu'])
            assert (campaign_root/'final-plan.json').read_text() == frozen
            campaign = json.loads((campaign_root/'campaign.json').read_text())
            campaign['deadline'] = 0
            atomic_json(campaign_root/'campaign.json', campaign)
            try:
                calibrate(spec_path, campaign_root, ['cpu'])
                raise AssertionError('Expired campaign silently reset its deadline')
            except TimeoutError:
                pass
        # Check actual checkpoint/resume for the non-default Shampoo recipe.
        c = Config(phase='verification', tiny=True, device='cpu', precision='fp32',
                   optimizer='shampoo', sequence=8, total_tokens=64, accumulation=1,
                   warmup_tokens=16, eval_tokens=64, eval_every=4, checkpoint_every=4,
                   momentum=.9, shampoo_graft='adagrad', fallback_beta2=.99,
                   fallback_weight_decay=.01, refresh=2)
        with patch('train.DATASET', 'local-fixture'):
            train(c, data, root/'straight')
            train(c, data, root/'resumed', stop_after_steps=3)
            train(c, data, root/'resumed', resume=True)
        a, _ = read_checkpoint(root/'straight')
        b, _ = read_checkpoint(root/'resumed')
        for key in ('model', 'optimizer', 'sampler', 'scheduler', 'fisher_rng'):
            assert_nested(unittest.TestCase(), a[key], b[key])
    print('Calibration checks passed: paired selection, shared fallback, failure handling, retention and recipe resume.')


if __name__ == '__main__':
    main()
