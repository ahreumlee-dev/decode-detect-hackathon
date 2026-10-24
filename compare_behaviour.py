"""Paired 80/20 ablation: identical rows and model, six features toggled."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'src'))
from behaviour_features import FEATURES

namespace = {}
book = json.loads((root/'src/starter_notebook.ipynb').read_text())
with threadpool_limits(limits=1):
    for cell in book['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'TRAIN_DIR = ' in source:
            source = source.replace('Path("../outputs")', 'Path('+repr(str(root/'outputs'))+')')
            source += '\nTRAIN_DIR = Path('+repr(str(args.train.resolve()))+')\n'
        if 'X_fit, X_check' in source:
            # Execute the actual notebook split and pipeline construction only.
            exec(source.split('model.fit(X_fit, y_fit)')[0], namespace)
            break
        exec(source, namespace)
    else:
        raise RuntimeError('Notebook split cell not found; review comparison runner.')
    X_fit, X_check = namespace['X_fit'], namespace['X_check']
    y_fit, y_check = namespace['y_fit'], namespace['y_check']
    hashes = namespace['train_features'].loc[X_check.index, 'TRANSACTION_HASH'].to_numpy()
    report = {'seed': 0, 'test_size': 0.2, 'fit_rows': len(X_fit), 'holdout_rows': len(X_check),
              'holdout_positive_rows': int(y_check.sum()),
              'holdout_hash_sha256': hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),
              'protocol': 'Offline full-train label-free features before stratified row holdout. Same pipeline and rows. Person 1 counts included in both.',
              'models': {}}
    for name, columns in [('without_behaviour', [c for c in X_fit if c not in FEATURES]),
                          ('with_behaviour', list(X_fit.columns))]:
        model = clone(namespace['model'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(X_fit[columns], y_fit)
        scores = model.predict_proba(X_check[columns])[:, 1]
        ap = average_precision_score(y_check, scores)
        k = int(np.ceil(len(scores)*0.01))
        order = np.lexsort((hashes, -scores))[:k]
        hits = int(y_check[order].sum())
        report['models'][name] = {'feature_count': len(columns), 'features': columns,
            'average_precision': float(ap), 'score_100_ap': float(100*ap),
            'top_1_percent_rows': k, 'precision_at_top_1_percent': hits/k,
            'recall_at_top_1_percent': hits/int(y_check.sum()),
            'convergence_warnings': [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]}
    report['delta_score_100_ap'] = report['models']['with_behaviour']['score_100_ap']-report['models']['without_behaviour']['score_100_ap']
    (root/'outputs/behaviour_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
