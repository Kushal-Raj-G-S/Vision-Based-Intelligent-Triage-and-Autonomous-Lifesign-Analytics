"""Cross-version joblib loading for the hybrid BPM correction models.

`models/bpm_correction_gbr.joblib` (and the per-fold bundles under
`models/cv_folds/`) were trained under scikit-learn 1.7.2. Newer sklearn
(1.9.0+) fails to unpickle them with `ModuleNotFoundError: No module named
'_loss'` — GradientBoostingRegressor's Cython loss objects were pickled with
a bare `_loss` module reference that only resolves as `sklearn._loss._loss`
in current sklearn's package layout. Aliasing that name in `sys.modules`
before unpickling is the standard workaround; it does not change what the
model computes, only how the old pickle's class references are resolved.
"""
import sys


def load_joblib_compat(path):
    import joblib
    import sklearn._loss._loss as _sklearn_loss_ext

    sys.modules.setdefault("_loss", _sklearn_loss_ext)
    return joblib.load(path)
