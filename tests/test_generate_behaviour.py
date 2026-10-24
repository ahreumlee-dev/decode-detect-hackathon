import pandas as pd
import pytest
from src import generate_behaviour as module
from src.behaviour_features import FEATURES


def test_generate_and_ml_handoff(tmp_path, monkeypatch):
    cleaned = tmp_path/'clean.parquet'
    output = tmp_path/'features.parquet'
    pd.DataFrame({'unused':[1]}).to_parquet(cleaned)
    payments = pd.DataFrame({'TRANSACTION_HASH':['a','b'], 'TIMESTAMP':[0,1], 'FROM_ADDRESS':['A','A'], 'TO_ADDRESS':['B','C'], 'PAID_USD':[2.,3.], 'RECEIVED_USD':[1.,2.]})
    monkeypatch.setattr(module, 'connect_cleaned_payments', lambda *args: payments)
    with pytest.raises(FileNotFoundError, match='04'):
        module.load_for_ml(output, cleaned, ['a','b'])
    features = module.generate(cleaned, tmp_path, output)
    assert list(features.columns) == ['TRANSACTION_HASH']+FEATURES
    assert len(module.load_for_ml(output, cleaned, ['b','a'])) == 2
    with pytest.raises(ValueError, match='hashes'):
        module.load_for_ml(output, cleaned, ['other'])
    pd.DataFrame({'changed':[2]}).to_parquet(cleaned)
    with pytest.raises(ValueError, match='stale'):
        module.load_for_ml(output, cleaned, ['a','b'])
