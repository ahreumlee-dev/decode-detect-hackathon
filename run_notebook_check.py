"""Execute integrated notebook code with explicit local dataset paths."""
import argparse
import json
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=Path, required=True)
parser.add_argument('--test', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'src'))
book = json.loads((root/'src/starter_notebook.ipynb').read_text())
namespace = {}
for cell in book['cells']:
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    if 'TRAIN_DIR = ' in source:
        source = source.replace('Path("../outputs")', repr(str(root/'outputs')))
        source = source.replace('OUTPUT_DIR = '+repr(str(root/'outputs')), 'OUTPUT_DIR = Path('+repr(str(root/'outputs'))+')')
        source += '\nTRAIN_DIR = Path(' + repr(str(args.train.resolve())) + ')\nTEST_DIR = Path(' + repr(str(args.test.resolve())) + ')\n'
    with threadpool_limits(limits=1):
        exec(compile(source, 'starter_notebook.ipynb', 'exec'), namespace)
pred = namespace['predictions']
assert len(pred) == len(namespace['test']) == 40101
assert pred.TRANSACTION_HASH.is_unique and pred.score.between(0, 1).all()
assert set(pred.TRANSACTION_HASH) == set(namespace['test'].TRANSACTION_HASH)
assert list(namespace['X'].columns) == list(namespace['test_features'].drop(columns='TRANSACTION_HASH').columns)
print('Verified test hashes, scores and train/test feature column alignment.')
