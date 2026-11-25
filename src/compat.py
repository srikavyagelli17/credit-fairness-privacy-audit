"""Import shims for library versions pinned in ``requirements.txt``."""

from __future__ import annotations

import functools
from typing import Any


def load_diffprivlib_logistic_regression() -> tuple[type, Any]:
    """Return diffprivlib's logistic regression and its privacy-leak warning.

    diffprivlib 0.6.6 imports tree symbols removed from scikit-learn 1.9 and
    passes ``multi_class`` into ``LogisticRegression``, which no longer accepts
    that argument. Patching those two import-time mismatches lets the logistic
    model run. This project does not use diffprivlib's tree models.
    """
    import numpy as np
    import sklearn.tree._tree as tree

    if not hasattr(tree, "DOUBLE"):
        tree.DOUBLE = np.float64
    if not hasattr(tree, "DTYPE"):
        tree.DTYPE = np.float32

    from sklearn.linear_model import LogisticRegression as SklearnLogistic

    if not getattr(SklearnLogistic, "_audit_multiclass_patched", False):
        original_init = SklearnLogistic.__init__

        @functools.wraps(original_init)
        def _init_without_multiclass(self: Any, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("multi_class", None)
            original_init(self, *args, **kwargs)

        SklearnLogistic.__init__ = _init_without_multiclass  # type: ignore[method-assign]
        SklearnLogistic._audit_multiclass_patched = True  # type: ignore[attr-defined]

    from diffprivlib.models.logistic_regression import LogisticRegression
    from diffprivlib.utils import PrivacyLeakWarning

    return LogisticRegression, PrivacyLeakWarning
