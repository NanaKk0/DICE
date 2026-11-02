from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from workflow_app.screen_io import LifecycleScreenIOAdapter, _StudentScreenScore
from workflow_backends.compact_score_store import write_compact_score_store
from workflow_core.lifecycle.compact_scores import CompactScoreView
from workflow_policies.gmm_reference_tail import GMMScoreResult


def test_compact_sparse_element_features_roundtrip_after_json_key_sort(tmp_path):
    ids = ('a', 'b', 'c')
    means = np.arange(6, dtype=np.float32).reshape(3, 2)
    scored = GMMScoreResult(np.array(ids), np.array([.1, .5, .9]), np.ones(3), means,
        {'width': 2}, {'O': (('b',), means[1:2]), 'H': (ids, means)})
    receipt = write_compact_score_store(tmp_path/'result', scored, model_digest='m', reference_digest='r')
    receipt = json.loads(json.dumps(receipt, sort_keys=True))
    assert len(json.dumps(receipt)) < 4096
    view = CompactScoreView(receipt, ordered_identities=ids, expected_model_digest='m', expected_reference_digest='r')
    retained = view.element_means
    view.close(cleanup=True)
    assert retained['O'][0] == ('b',)
    np.testing.assert_array_equal(retained['O'][1], means[1:2])
    assert not (tmp_path/'result').exists()


def test_phased_readout_accepts_compact_lease_and_cleans_it(tmp_path):
    from workflow_backends.student.readout import _phase_score
    from workflow_policies.gmm_reference_tail import GMMReferenceFitResult
    schema = {'width': 2}
    scored = GMMScoreResult(np.array(['a']), np.array([.5]), np.array([1.]), np.ones((1, 2), dtype=np.float32), schema,
        {'H': (('a',), np.ones((1, 2), dtype=np.float32))})
    receipt = write_compact_score_store(tmp_path/'result', scored, model_digest='m', reference_digest='r')
    view = CompactScoreView(receipt, ordered_identities=('a',), expected_model_digest='m', expected_reference_digest='r')
    fit = GMMReferenceFitResult(tmp_path/'router', 'gmm-reference-tail', {'H': 1}, schema)
    outcome = _phase_score(SimpleNamespace(emit_frame_means=True), ('a',), fit, view)
    assert outcome.values == {'a': .5}
    retained = outcome.element_frame_means['H']['a']
    outcome.close()
    np.testing.assert_array_equal(retained, [1, 1])
    assert not view.root.exists()


def reuse_fixture(tmp_path, monkeypatch):
    import workflow_app.screen_io as module
    candidates = [Atoms('H'), Atoms('HO')]
    for frame, identity in zip(candidates, ('a', 'b')):
        frame.info['structure_id'] = identity
    background = Atoms('HO')
    background.info['structure_id'] = 'ref'
    monkeypatch.setattr(module, 'read_extxyz_structures', lambda *a, **k: [background])
    schema = {'width': 2}
    calls = []
    full = SimpleNamespace(
        candidate_vectors={'a': np.array([1., 2.]), 'b': np.array([3., 4.])},
        background_vectors={'ref': np.array([5., 6.])}, descriptor_schema=schema,
        candidate_element_vectors={'H': {'a': np.array([1., 2.]), 'b': np.array([3., 4.])}, 'O': {'b': np.array([4., 5.])}},
        background_element_vectors={'H': {'ref': np.array([5., 6.])}, 'O': {'ref': np.array([7., 8.])}})
    def vectors(**kwargs):
        calls.append(tuple(frame.info['structure_id'] for frame in kwargs['candidate_frames']))
        return full
    student = SimpleNamespace(resolve_model_expectations=lambda *a, **k: (SimpleNamespace(path=tmp_path/'model'),), _read_structure_vectors=vectors)
    adapter = LifecycleScreenIOAdapter(SimpleNamespace(root=tmp_path, iter_index=0, student={'members': 1, 'type_map': ['H', 'O']}),
        None, student, 2, None, None, None, None, None)
    score = _StudentScreenScore({'a': .5, 'b': .3}, full.candidate_vectors, schema,
        full.candidate_element_vectors, 'iter0/r1/train/student', 'iter0/r1/train/dataset')
    class Source:
        def replay(self, *a, **k):
            raise AssertionError('reused candidate features must not replay geometry')
    options = dict(frames=candidates, source=Source(), work_dir=tmp_path/'vectors', report=None, round_index=1)
    return adapter, score, options, calls, full


def test_reuse_reads_only_background_and_preserves_all_selector_vectors(tmp_path, monkeypatch):
    adapter, score, options, calls, full = reuse_fixture(tmp_path, monkeypatch)
    result = adapter.read_student_vectors(**options, reused_score=score)
    assert calls == [()]
    for element, rows in full.candidate_element_vectors.items():
        for identity, row in rows.items():
            np.testing.assert_array_equal(result.candidate_element_vectors[element][identity], row)
    assert tuple(result.candidate_vectors) == ('a', 'b')
    assert tuple(result.background_vectors) == ('ref',)
    assert result.model_identity == score.model_identity
    assert result.dataset_identity == score.dataset_identity


@pytest.mark.parametrize('fault', ['model', 'dataset', 'schema', 'order', 'presence', 'width'])
def test_reuse_rejects_stale_or_incomplete_evidence(tmp_path, monkeypatch, fault):
    adapter, score, options, calls, full = reuse_fixture(tmp_path, monkeypatch)
    if fault == 'model':
        score = replace(score, model_identity='old-model')
    elif fault == 'dataset':
        score = replace(score, dataset_identity='old-dataset')
    elif fault == 'schema':
        score = replace(score, descriptor_schema={'width': 3})
    elif fault == 'order':
        score = replace(score, frame_means=dict(reversed(list(score.frame_means.items()))))
    elif fault == 'presence':
        score = replace(score, element_frame_means={'H': score.element_frame_means['H']})
    else:
        score = replace(score, element_frame_means={'H': {'a': [1], 'b': [1, 2]}, 'O': {'b': [1, 2]}})
    with pytest.raises(ValueError, match='reused student'):
        adapter.read_student_vectors(**options, reused_score=score)


def test_invalid_compact_evidence_closes_every_opened_mmap(tmp_path, monkeypatch):
    scored = GMMScoreResult(np.array(['a']), np.array([.5]), np.array([np.nan]),
        np.ones((1, 2), dtype=np.float32), {'width': 2},
        {'H': (('a',), np.ones((1, 2), dtype=np.float32))})
    receipt = write_compact_score_store(tmp_path/'invalid', scored, model_digest='m', reference_digest='r')
    original = np.load
    opened = []
    def load(*args, **kwargs):
        array = original(*args, **kwargs)
        opened.append(array)
        return array
    monkeypatch.setattr(np, 'load', load)
    with pytest.raises(ValueError, match='statistics'):
        CompactScoreView(receipt, ordered_identities=('a',), expected_model_digest='m', expected_reference_digest='r')
    assert opened and all(array._mmap.closed for array in opened)


def test_fused_feature_bridge_reads_ordinal_arrays_without_population_lookups(tmp_path):
    from collections.abc import Mapping
    from workflow_backends.student.readout import _fused_score
    from workflow_policies.gmm_reference_tail import GMMReferenceFitResult
    ids = tuple(f'id-{i}' for i in range(1000))
    class Ordinals(Mapping):
        def __iter__(self):
            return iter(ids)
        def __len__(self):
            return len(ids)
        def __getitem__(self, key):
            raise AssertionError('full population must use ordinal arrays')
    schema = {'width': 2}
    probabilities = np.full(1000, .5)
    rows = np.ones((1000, 2), dtype=np.float32)
    score = SimpleNamespace(identities=ids, values=Ordinals(), means=Ordinals(),
        frame_p=probabilities, frame_means=rows, descriptor_schema=schema,
        close=lambda **kwargs: None)
    fit = GMMReferenceFitResult(tmp_path/'router', 'gmm-reference-tail', {'H': 1}, schema)
    result = _fused_score(SimpleNamespace(emit_frame_means=True), ids, SimpleNamespace(fit=fit, score=score))
    assert result.values['id-999'] == .5
    np.testing.assert_array_equal(result.frame_means['id-999'], [1., 1.])
    result.close()
