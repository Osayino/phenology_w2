"""Verify a completed notebook and its saved full-data experiment artifacts."""
from pathlib import Path
from datetime import date
import hashlib
import json
import sys

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'casework'))
from knn_utils import uniform_predictions


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


results = json.loads((ROOT / 'results/experiment_results.json').read_text(encoding='utf-8'))
notebook = json.loads((ROOT / 'casework/case_notebook.ipynb').read_bytes())
code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
assert [c['execution_count'] for c in code_cells] == list(range(1, len(code_cells) + 1))
assert not [o for c in code_cells for o in c['outputs'] if o.get('output_type') == 'error']
assert any(sys.executable.lower() in ''.join(o.get('text', [])).lower()
           for c in code_cells for o in c['outputs'] if o.get('output_type') == 'stream')

extension_summary = {'status': 'not_present'}
with np.load(ROOT / 'results/pixel_dataset.npz', allow_pickle=False) as data, \
     np.load(ROOT / 'results/predictions.npz', allow_pickle=False) as predictions:
    train, val = data['train_indices'], data['val_indices']
    y_trainval, y_test = data['y_trainval'], data['y_test']
    assert data['X_trainval'].shape == (13745, 111)
    assert data['X_test'].shape == (14105, 111)
    assert (len(train), len(val), len(y_test)) == (10996, 2749, 14105)
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(13745))
    np.testing.assert_array_equal(data['doys'], np.arange(242, 279))
    np.testing.assert_array_equal(data['ids_trainval'][:, 0], y_trainval)
    np.testing.assert_array_equal(data['ids_test'][:, 0], y_test)
    coordinates = np.vstack([data['ids_trainval'][:, 2:], data['ids_test'][:, 2:]])
    assert len(np.unique(coordinates, axis=0)) == 27850
    for key in ('train_indices', 'val_indices', 'ids_test'):
        np.testing.assert_array_equal(predictions[key], data[key])
    np.testing.assert_array_equal(predictions['y_test'], y_test)
    for prefix in ('raw', 'rp'):
        val_neighbors = predictions[f'{prefix}_val_neighbors']
        test_neighbors = predictions[f'{prefix}_test_neighbors']
        assert np.isin(val_neighbors, train).all() and np.isin(test_neighbors, train).all()
        scores = [accuracy_score(y_trainval[val], uniform_predictions(val_neighbors, y_trainval, k))
                  for k in results['k_candidates']]
        np.testing.assert_array_equal(scores, results[f'{prefix}_validation_accuracy'])
        best_k = results['k_candidates'][int(np.argmax(scores))]
        assert best_k == results[f'{prefix}_best_k']
        predicted = predictions[f'{prefix}_prediction']
        np.testing.assert_array_equal(predicted, uniform_predictions(test_neighbors, y_trainval, best_k))
        expected = {
            'Accuracy': accuracy_score(y_test, predicted),
            'Precision (macro)': precision_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
            'F1-score (macro)': f1_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
        }
        assert expected == results[f'{prefix}_test_metrics']
        matrix = confusion_matrix(y_test, predicted, labels=np.arange(6))
        np.testing.assert_array_equal(matrix, results[f'{prefix}_confusion_matrix'])
        np.testing.assert_array_equal(matrix.sum(axis=1), np.bincount(y_test, minlength=6))

    cache_hashes = {}
    for name, key, feature_key in zip(('rp_trainval', 'rp_test'), results['recurrence_cache'], ('X_trainval', 'X_test')):
        array_path = ROOT / 'cache' / f'{name}_{key}.npy'
        info = json.loads(array_path.with_suffix('.json').read_text(encoding='utf-8'))
        assert info['cache_key'] == key
        X = data[feature_key]
        digest = hashlib.sha256(json.dumps(info['recipe'], sort_keys=True).encode('utf-8'))
        for start in range(0, len(X), 1024):
            digest.update(np.ascontiguousarray(X[start:start + 1024]).tobytes())
        assert digest.hexdigest() == key
        assert sha256(array_path) == info['file_sha256']
        mapped = np.load(array_path, mmap_mode='r', allow_pickle=False)
        assert mapped.shape == (len(X), 58800) and mapped.dtype == np.uint8
        cache_hashes[name] = info['file_sha256']
        mapped._mmap.close()

    extension_paths = [ROOT / 'results' / name for name in
                       ('extension_results.json', 'extension_predictions.npz',
                        'extension_comparison.png')]
    if any(path.exists() for path in extension_paths):
        assert all(path.is_file() for path in extension_paths), 'Incomplete optional-extension artifacts'
        extension = json.loads(extension_paths[0].read_text(encoding='utf-8'))
        candidates = extension['validation_candidates']
        declared_names = list(extension['candidate_models'])
        assert candidates and [row['name'] for row in candidates] == declared_names
        assert len(declared_names) == len(set(declared_names))
        assert all(np.isfinite(row['accuracy']) and 0 <= row['accuracy'] <= 1
                   and np.isfinite(row['macro_f1']) and 0 <= row['macro_f1'] <= 1
                   for row in candidates)
        # max retains the first declared candidate when validation accuracies tie.
        selected = max(candidates, key=lambda row: row['accuracy'])
        assert extension['selected_model'] == selected['name']
        assert extension['samples'] == results['samples']
        assert extension['seed'] == results['seed']
        with np.load(extension_paths[1], allow_pickle=False) as extra:
            for key in ('train_indices', 'val_indices', 'ids_test'):
                np.testing.assert_array_equal(extra[key], data[key])
            np.testing.assert_array_equal(extra['y_test'], y_test)
            predicted = extra['prediction']
            assert predicted.shape == y_test.shape and np.isin(predicted, np.arange(6)).all()
            expected = {
                'Accuracy': accuracy_score(y_test, predicted),
                'Precision (macro)': precision_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
                'Recall (macro)': recall_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
                'F1-score (macro)': f1_score(y_test, predicted, labels=np.arange(6), average='macro', zero_division=0),
            }
            assert expected == extension['test_metrics']
            matrix = confusion_matrix(y_test, predicted, labels=np.arange(6))
            np.testing.assert_array_equal(matrix, extension['confusion_matrix'])
            np.testing.assert_array_equal(matrix.sum(axis=1), np.bincount(y_test, minlength=6))
            change_pp = 100 * (expected['Accuracy'] - results['raw_test_metrics']['Accuracy'])
            assert change_pp == extension['accuracy_change_percentage_points']
        assert extension_paths[2].stat().st_size > 1000
        extension_summary = {
            'status': 'passed', 'selected_model': selected['name'],
            'validation_accuracy': selected['accuracy'], 'validation_macro_f1': selected['macro_f1'],
            'test_metrics': expected, 'accuracy_change_percentage_points': change_pp,
            'checks': ['Selection agrees with saved validation scores and declared candidate order',
                       'Same test identities, labels and train/validation indices as baseline',
                       'Test metrics, confusion matrix and accuracy change independently recomputed',
                       'Extension comparison figure present'],
        }

for name in ('pixel_examples.png', 'comparison.png', 'confusion_matrices.png'):
    assert (ROOT / 'results' / name).stat().st_size > 1000

comparison_summary = {'status': 'not_present'}
comparison_paths = [ROOT / 'results' / name for name in
                    ('model_comparison.json', 'model_comparison_predictions.npz',
                     'model_comparison_selection.json')]
if any(path.exists() for path in comparison_paths):
    assert all(path.is_file() for path in comparison_paths), 'Incomplete optional-comparison artifacts'
    # This verifier reads predictions and the selection record, not fitted models.
    sys.path.insert(0, str(ROOT))
    from experiments.verify_comparison import verify_results as verify_model_comparison
    comparison_summary = verify_model_comparison(ROOT)

summary = {
    'status': 'passed', 'verification_date': date.today().isoformat(),
    'command': r'.\.venv\Scripts\python.exe checks\verify_results.py',
    'code_cells_with_sequential_outputs': len(code_cells), 'notebook_errors': 0,
    'samples': results['samples'], 'all_selected_pixels': 27850,
    'checks': ['Sequential saved execution counts', 'Feature/identity/label alignment',
               'Complete disjoint shared split', 'Validation-only K selection',
               'Predictions recomputed from saved training-neighbour indices',
               'Metrics and confusion matrices independently recomputed',
               'Matching full test supports', 'Cache input keys and full file checksums'],
    'code_sha256': hashlib.sha256('\n'.join(''.join(c['source']) for c in code_cells).encode('utf-8')).hexdigest(),
    'helper_sha256': {name: sha256(ROOT / 'casework' / name) for name in ('phenology_data.py', 'knn_utils.py')},
    'recurrence_file_sha256': cache_hashes,
    'numpy_version': np.__version__, 'sklearn_version': sklearn.__version__,
    'optional_extension': extension_summary,
    'optional_model_comparison': comparison_summary,
    'pending': ['Full rubric review', 'Student review of explanations and reflection'],
}
(ROOT / 'results/verification.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
