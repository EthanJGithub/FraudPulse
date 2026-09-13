import numpy as np
import pandas as pd
from src.ml import train as module


def test_final_test_never_selects_iterations_or_threshold(tmp_path, monkeypatch):
    calls = {}
    class Model:
        def __init__(self, **kw): pass
        def fit(self, x, y, eval_set, **kw):
            calls['training'] = set(x.index)
            calls['validation'] = set(eval_set[0][0].index)
        def predict_proba(self, x):
            calls.setdefault('predictions', []).append(set(x.index))
            p = np.linspace(.1, .9, len(x))
            return np.column_stack([1-p, p])
    class Isolation:
        def __init__(self, **kw): pass
        def fit(self, x): pass
        def score_samples(self, x): return np.zeros(len(x))
    def threshold(y, scores):
        assert calls['predictions'] == [calls['validation']]
        return .5, .2
    monkeypatch.setattr(module.xgb, 'XGBClassifier', Model)
    monkeypatch.setattr(module, 'IsolationForest', Isolation)
    monkeypatch.setattr(module, '_best_threshold', threshold)
    monkeypatch.setattr(module.joblib, 'dump', lambda *a: None)
    df = pd.DataFrame({'Class': [0, 1]*100, 'Amount': np.arange(200), 'Time': np.arange(200)})
    result = module.train(df=df, model_dir=str(tmp_path))
    final_test = calls['predictions'][1]
    assert not final_test & calls['training']
    assert not final_test & calls['validation']
    assert not calls['training'] & calls['validation']
    assert len(final_test | calls['training'] | calls['validation']) == len(df)
    assert result['n_validation'] == result['n_test'] == 40
