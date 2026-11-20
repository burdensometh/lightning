"""
==========================================================
Multi-label classification with partially observed labels
==========================================================

In multi-label annotation campaigns a (sample, class) pair is often
*not audited*: its true label is unknown.  An unknown label is not a
negative label.  Encoding it as a negative teaches the binary classifier
for that class that a sample is not a member, although nobody ever
checked.  It also biases the class prevalence estimate.

:class:`MultiLabelClassifier` is a training layer on top of the binary
linear classifiers shipped with lightning.  It fits one binary linear
classifier per class, and the sample x class observation mask defines the
*training population* of each of those classifiers:

* a known positive is a ``1``;
* a known negative is a ``0``;
* an unknown (or unaudited) cell is never encoded as a negative and never
  contributes to that class's loss.

Consequences of this design:

* Different classes may have very different support (number of observed
  samples and number of positives).  Each per-class model is fitted on its
  own observed rows only, so a class with sparse support does not make the
  other classes lose data, and the other classes do not dilute it.
* The class vocabulary is frozen at fit time (``classes_``).  A class with
  zero observed labels does not disappear from the model; it keeps its slot
  on the prediction axis.
* ``predict``, ``predict_proba`` and ``decision_function`` always return the
  full class axis, in the order of ``classes_``.
* Every class slot follows the same field contract: ``classes_``,
  ``coef_``, ``intercept_``, ``decision_function``, ``predict_proba``,
  ``predict``, plus bookkeeping fields (``kind_``, ``n_observed_``,
  ``n_positive_``).  A class that cannot be learned is represented by an
  explicit constant prior model that implements exactly the same contract
  (see ``on_unobserved`` and ``on_single_class``), or the fit is rejected.
* The observation mask is consumed by this training layer only.  It is never
  appended to ``X``, never passed to the per-class estimators, and is not an
  input feature at prediction time.

Usage
-----
``y`` is a 2-dimensional indicator matrix with shape
``(n_samples, n_classes)``.  Unknown cells can be expressed in two ways:

1. explicitly, via ``sample_mask`` (``True`` means *observed/audited*);
2. implicitly, with ``NaN`` (dense float ``y``) or with the value of
   ``unknown_label``.

.. code-block:: python

    from lightning.classification import MultiLabelClassifier
    from lightning.classification import SGDClassifier

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=["sports", "politics", "tech"],
    )
    clf.fit(X, y, sample_mask=M)      # M[i, k] == True iff y[i, k] audited

    proba = clf.predict_proba(X)      # (n_samples, 3), full class axis
    pred = clf.predict(X)             # (n_samples, 3), 0/1
    acc = clf.score(X, y, sample_mask=M)   # only audited cells are scored

Notes
-----
* ``score`` and ``per_class_score`` evaluate on exactly the observed cells of
  the mask.  Rows containing at least one unknown label are kept: their
  audited cells are scored, their unaudited cells are skipped.  Whole rows
  are never dropped.
* The observation mask is a target-side object and is not a feature.  The
  number of features seen by the layer (``n_features_in_``) is exactly
  ``X.shape[1]``.
* The per-class estimator must be a *binary linear* classifier exposing
  ``classes_ == [0, 1]``, ``coef_`` with shape ``(1, n_features)`` and
  either ``decision_function`` or ``predict_proba``.  Missing
  ``intercept_`` is treated as zero.  This is what makes the uniform field
  contract (and the layer-level ``coef_`` matrix) possible.
* The layer is not a scikit-learn 1-dimensional ``fit(X, y)`` estimator:
  it deliberately expects a 2-dimensional target and a mask, so generic
  cross-validation helpers that only know about 1-dimensional targets should
  not be used directly.
"""

# Author: lightning contributors
# License: BSD

import numpy as np
import scipy.sparse as sp
import scipy.special

from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseEstimator

_EPS = 1e-6
_POLICIES = ("prior", "raise")


def _expit(x):
    return scipy.special.expit(x)


def _logit(p):
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _isnan_like(arr):
    """Boolean mask of NaN/None values, for float and object arrays."""
    if arr.dtype.kind in "fc":
        return np.isnan(arr)
    if arr.dtype.kind == "O":
        out = np.empty(arr.shape, dtype=bool)
        flat_in = arr.ravel()
        flat_out = out.ravel()
        for i, value in enumerate(flat_in):
            if value is None:
                flat_out[i] = True
                continue
            try:
                flat_out[i] = bool(value != value)
            except (TypeError, ValueError):
                flat_out[i] = False
        return out
    return np.zeros(arr.shape, dtype=bool)


def _align_columns(obj, target, name):
    """Align a DataFrame-like object to the frozen class vocabulary.

    Objects without a ``columns`` attribute are returned unchanged: plain
    arrays are positional and validated against ``target`` by their width.
    """
    columns = getattr(obj, "columns", None)
    if columns is None:
        return obj

    columns = list(columns)
    target = list(target)

    if columns == target:
        return obj

    if len(columns) == len(target) and set(columns) == set(target):
        try:
            return obj[target]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Unable to reorder %s columns %r to the frozen class "
                "vocabulary %r." % (name, columns, target)
            ) from exc

    raise ValueError(
        "%s columns %r do not match the frozen class vocabulary %r."
        % (name, columns, target)
    )


def _as_bool_mask(mask, shape, name="sample_mask"):
    """Convert a dense or sparse 0/1 mask to a dense boolean array."""
    if sp.issparse(mask):
        if mask.shape != shape:
            raise ValueError(
                "%s has shape %r, expected %r."
                % (name, mask.shape, shape)
            )
        mask = mask.tocsr()
        if mask.data.size and mask.data.dtype.kind != "b":
            if not np.all(np.isin(mask.data, (0, 1))):
                raise ValueError(
                    "%s must contain only 0/1 (True means observed)." % name
                )
        return np.asarray(mask.astype(bool).toarray(), dtype=bool)

    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(
            "%s has shape %r, expected %r." % (name, arr.shape, shape)
        )

    if arr.dtype.kind == "b":
        return arr.astype(bool, copy=False)

    if arr.dtype.kind not in "iuf":
        raise ValueError(
            "%s must be a boolean or 0/1 array (True means observed)." % name
        )

    if not np.all(np.isin(arr, (0, 1))):
        raise ValueError(
            "%s must contain only 0/1 (True means observed)." % name
        )

    return arr.astype(bool)


def _target_shape(y):
    if sp.issparse(y):
        return y.shape
    if hasattr(y, "shape"):
        return tuple(y.shape)
    return np.asarray(y).shape


def _column_values(y, n_samples, k):
    """Return column ``k`` of ``y`` as a dense float array."""
    if sp.issparse(y):
        dense = y[:, [k]].todense()
        return np.asarray(dense, dtype=np.float64).ravel()
    return np.asarray(y)[:, k]


def _subset_rows(X, mask):
    """Row-subset a dense array, a DataFrame or a sparse matrix."""
    if sp.issparse(X):
        return X.tocsr()[mask]
    return X[mask]


class _ClassModel(object):
    """Uniform per-class model: fitted estimator *or* explicit prior.

    Whether the model was fitted or is a constant prior fallback, it exposes
    the same fields so that every class on the prediction axis obeys one
    contract.

    Fields
    ------
    classes_ : ndarray, shape (2,)
        Always ``[0, 1]`` (binary indicator semantics).
    coef_ : ndarray, shape (1, n_features)
    intercept_ : ndarray, shape (1,)
    decision_function(X) : ndarray, shape (n_samples,)
    predict_proba(X) : ndarray, shape (n_samples, 2)
    predict(X, threshold=0.5) : ndarray, shape (n_samples,)
    kind_ : {'fitted', 'single_class_prior', 'unobserved_prior'}
    n_observed_, n_positive_ : int
    prior_ : float or None
        Constant prior for fallback models, None for fitted models.
    estimator_ : estimator or None
        The raw fitted estimator for fitted models, None for fallbacks.
    """

    def __init__(self, estimator, kind, n_features, n_observed, n_positive,
                 prior=None):
        self.estimator_ = estimator
        self.kind_ = kind
        self.n_observed_ = int(n_observed)
        self.n_positive_ = int(n_positive)
        self.classes_ = np.array([0, 1], dtype=np.int32)

        if estimator is None:
            # The effective prior is clipped away from 0 and 1 so that the
            # fallback is still a proper probability model.
            prior = float(np.clip(prior, _EPS, 1.0 - _EPS))
            self.prior_ = prior
            self.coef_ = np.zeros((1, n_features), dtype=np.float64)
            self.intercept_ = np.array([_logit(prior)], dtype=np.float64)
            self._native_proba = False
        else:
            self.prior_ = None
            coef = np.asarray(estimator.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            self.coef_ = np.array(coef, dtype=np.float64, copy=True)
            intercept = getattr(estimator, "intercept_", None)
            if intercept is None:
                self.intercept_ = np.zeros(1, dtype=np.float64)
            else:
                intercept = np.asarray(intercept, dtype=np.float64).ravel()
                self.intercept_ = intercept.copy()
            # ``hasattr`` resolves lightning's ``predict_proba`` property,
            # which raises AttributeError for losses without probabilities.
            self._native_proba = hasattr(estimator, "predict_proba")

    def decision_function(self, X):
        return safe_sparse_dot(X, self.coef_.T).ravel() + self.intercept_[0]

    def predict_proba(self, X):
        if self._native_proba:
            proba = np.asarray(self.estimator_.predict_proba(X),
                               dtype=np.float64)
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise ValueError(
                    "predict_proba of a per-class estimator must return "
                    "shape (n_samples, 2), got %r." % (proba.shape,)
                )
            pos = np.flatnonzero(np.asarray(self.estimator_.classes_) == 1)
            if pos.size != 1:
                raise ValueError(
                    "Per-class estimators must be binary with "
                    "classes_ == [0, 1]."
                )
            positive = proba[:, pos[0]]
        else:
            positive = _expit(self.decision_function(X))

        out = np.empty((positive.shape[0], 2), dtype=np.float64)
        out[:, 1] = positive
        out[:, 0] = 1.0 - positive
        return out

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype(np.int32)


class MultiLabelClassifier(ClassifierMixin, BaseEstimator):
    r"""Multi-label linear classifier with a per-class observation mask.

    One binary linear classifier is fitted per class.  For class ``k``, only
    the rows where ``sample_mask[:, k]`` is True take part in fitting, with
    the observed values of ``y[:, k]`` as targets.  Unobserved cells are
    neither positives nor negatives and are never shown to the model.

    Parameters
    ----------
    base_estimator : estimator or None, default=None
        Binary linear classifier exposing ``classes_ == [0, 1]``,
        ``coef_`` with shape ``(1, n_features)`` and either
        ``decision_function`` or ``predict_proba``.  It is cloned for every
        class.  If None, ``SGDClassifier(loss="log", random_state=0)`` from
        :mod:`lightning.classification` is used.

    classes : array-like or None, default=None
        Frozen class vocabulary (column axis labels).  When None, the
        vocabulary is inferred once at the first ``fit`` from the target
        columns (DataFrame columns if available, positional indices
        otherwise) and then frozen.  Subsequent fits must use the same
        vocabulary.

    on_unobserved : {'prior', 'raise'}, default='prior'
        What to do with a class that has no observed labels at all.
        ``'prior'`` builds an explicit constant model predicting
        ``unobserved_prior``.  ``'raise'`` rejects the fit with
        ``ValueError``.

    on_single_class : {'prior', 'raise'}, default='prior'
        What to do with a class whose observed labels contain only one value
        (all 0 or all 1).  ``'prior'`` builds an explicit constant model
        predicting that observed value (probability clipped to
        ``[eps, 1 - eps]``).  ``'raise'`` rejects the fit.

    unobserved_prior : float, default=0.5
        Probability predicted by the fallback model of a class with no
        observed labels (only used with ``on_unobserved='prior'``).

    threshold : float, default=0.5
        Probability threshold used by :meth:`predict`: a cell is positive
        when its probability is strictly greater than ``threshold``.

    unknown_label : scalar or None, default=None
        Sentinel value used in a dense ``y`` to mark unknown cells, in
        addition to ``NaN`` (which is always treated as unknown).  Cells
        marked observed by ``sample_mask`` must not use the sentinel.

    Attributes
    ----------
    classes_ : ndarray
        Frozen class vocabulary.  Prediction outputs always use this full
        axis, even for classes that were never observed.

    n_features_in_ : int
        Number of features seen during fit.  It is exactly ``X.shape[1]``:
        the observation mask is not a feature.

    coef_ : ndarray, shape (n_classes, n_features)
        Assembled coefficient matrix.  Rows of fallback classes are zero.

    intercept_ : ndarray, shape (n_classes,)
        Assembled intercepts.  Fallback classes hold the log-odds of their
        constant prior.

    estimators_ : list of length n_classes
        Per-class :class:`_ClassModel` objects, all obeying the same field
        contract.  Use ``estimators_[k].kind_`` to distinguish fitted models
        from prior fallbacks and ``estimators_[k].estimator_`` to reach the
        raw fitted estimator.

    model_kind_ : ndarray of str, shape (n_classes,)
        ``'fitted'``, ``'single_class_prior'`` or ``'unobserved_prior'``.

    n_observed_ : ndarray, shape (n_classes,)
        Number of audited samples used by each class.

    n_positives_ : ndarray, shape (n_classes,)
        Number of observed positives per class.

    class_priors_ : ndarray, shape (n_classes,)
        Observed prevalence per class, NaN for classes with no observations.
    """

    def __init__(self, base_estimator=None, classes=None,
                 on_unobserved="prior", on_single_class="prior",
                 unobserved_prior=0.5, threshold=0.5, unknown_label=None):
        self.base_estimator = base_estimator
        self.classes = classes
        self.on_unobserved = on_unobserved
        self.on_single_class = on_single_class
        self.unobserved_prior = unobserved_prior
        self.threshold = threshold
        self.unknown_label = unknown_label

    # ------------------------------------------------------------------
    # validation helpers
    # ------------------------------------------------------------------
    def _validate_params(self):
        if self.on_unobserved not in _POLICIES:
            raise ValueError(
                "on_unobserved must be one of %r, got %r."
                % (_POLICIES, self.on_unobserved)
            )
        if self.on_single_class not in _POLICIES:
            raise ValueError(
                "on_single_class must be one of %r, got %r."
                % (_POLICIES, self.on_single_class)
            )
        if not 0.0 <= float(self.unobserved_prior) <= 1.0:
            raise ValueError("unobserved_prior must be in [0, 1].")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1].")

    def _make_base_estimator(self):
        if self.base_estimator is None:
            from .sgd import SGDClassifier
            return SGDClassifier(loss="log", random_state=0)
        return clone(self.base_estimator)

    def _resolve_classes(self, y):
        """Resolve and freeze the class vocabulary."""
        shape = _target_shape(y)
        if len(shape) != 2:
            raise ValueError(
                "y must be a 2-dimensional (n_samples, n_classes) target, "
                "got shape %r. MultiLabelClassifier does not accept 1-D "
                "targets." % (shape,)
            )

        n_columns = shape[1]
        fitted = hasattr(self, "classes_")

        if self.classes is not None:
            target = np.asarray(self.classes)
            if target.ndim != 1 or target.size == 0:
                raise ValueError("classes must be a non-empty 1-D array.")
            if len(np.unique(target)) != target.size:
                raise ValueError("classes must not contain duplicates.")
            if fitted and not np.array_equal(target, self.classes_):
                raise ValueError(
                    "The class vocabulary is frozen: this estimator was "
                    "fitted with %r, but classes=%r was given."
                    % (list(self.classes_), list(target))
                )
            if fitted:
                if n_columns != len(self.classes_):
                    raise ValueError(
                        "The class vocabulary is frozen with %d classes "
                        "(%r); got a target with %d columns."
                        % (len(self.classes_), list(self.classes_),
                           n_columns)
                    )
                return self.classes_
            if target.size != n_columns:
                raise ValueError(
                    "classes has %d entries but y has %d columns."
                    % (target.size, n_columns)
                )
            return target

        if fitted:
            if n_columns != len(self.classes_):
                raise ValueError(
                    "The class vocabulary is frozen with %d classes (%r); "
                    "got a target with %d columns."
                    % (len(self.classes_), list(self.classes_), n_columns)
                )
            return self.classes_

        columns = getattr(y, "columns", None)
        if columns is not None:
            target = np.asarray(columns)
            if len(np.unique(target)) != target.size:
                raise ValueError("Target columns must not contain duplicates.")
            return target

        return np.arange(n_columns)

    def _validate_fitted_estimator(self, estimator, n_features, class_label):
        classes = getattr(estimator, "classes_", None)
        if classes is None or not np.array_equal(
                np.sort(np.asarray(classes)), np.array([0, 1])):
            raise ValueError(
                "The base estimator for class %r must be binary with "
                "classes_ == [0, 1]; got classes_=%r."
                % (class_label, classes)
            )

        coef = getattr(estimator, "coef_", None)
        if coef is None:
            raise ValueError(
                "The base estimator for class %r must expose coef_."
                % (class_label,)
            )
        coef = np.asarray(coef)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.ndim != 2 or coef.shape != (1, n_features):
            raise ValueError(
                "The base estimator for class %r must be a linear binary "
                "classifier with coef_ of shape (1, n_features); got %r "
                "with n_features=%d. Kernel and multiclass estimators are "
                "not supported by the layer." % (class_label, coef.shape,
                                                 n_features)
            )

        if not (hasattr(estimator, "decision_function") or
                hasattr(estimator, "predict_proba")):
            raise ValueError(
                "The base estimator for class %r must expose "
                "decision_function or predict_proba." % (class_label,)
            )

    # ------------------------------------------------------------------
    # target handling
    # ------------------------------------------------------------------
    def _prepare_targets(self, y, sample_mask, n_samples, target):
        """Validate y/mask and return (y_repr, dense bool observation mask)."""
        shape = (n_samples, len(target))

        y = _align_columns(y, target, "y")
        if hasattr(sample_mask, "columns"):
            sample_mask = _align_columns(sample_mask, target, "sample_mask")

        raw_shape = _target_shape(y)
        if len(raw_shape) != 2 or tuple(raw_shape) != shape:
            raise ValueError(
                "y must be a 2-dimensional (n_samples, n_classes) target "
                "of shape %r, got %r."
                % (shape, raw_shape)
            )

        if sp.issparse(y):
            if self.unknown_label is not None:
                raise ValueError(
                    "unknown_label cannot be used with a sparse y: a sparse "
                    "indicator target has implicit zeros and no way to carry "
                    "a separate sentinel. Use sample_mask instead."
                )
            y_repr = y.tocsr()
            y_repr.sum_duplicates()
            if y_repr.data.size:
                if np.isnan(y_repr.data).any():
                    raise ValueError(
                        "Sparse y contains NaN. Mark unaudited cells with "
                        "sample_mask rather than NaN in a sparse target."
                    )
                if not np.all(np.isin(y_repr.data, (0, 1))):
                    raise ValueError(
                        "Sparse y must contain only 0/1 indicator values. "
                        "Use sample_mask to mark unknown cells."
                    )
            if sample_mask is None:
                mask = np.ones(shape, dtype=bool)
            else:
                mask = _as_bool_mask(sample_mask, shape)
            return y_repr, mask

        y_repr = np.asarray(y)
        nan_mask = _isnan_like(y_repr)
        if self.unknown_label is None:
            unknown = nan_mask
        else:
            try:
                with np.errstate(invalid="ignore"):
                    sentinel = np.asarray(y_repr == self.unknown_label)
            except (TypeError, ValueError):
                sentinel = np.zeros(shape, dtype=bool)
            unknown = nan_mask | np.asarray(sentinel, dtype=bool)

        if sample_mask is None:
            mask = ~unknown
        else:
            mask = _as_bool_mask(sample_mask, shape)
            conflict = mask & unknown
            if conflict.any():
                rows, cols = np.nonzero(conflict)
                raise ValueError(
                    "Cells marked observed by sample_mask contain NaN "
                    "or unknown_label (first at row %d, class column %d)."
                    % (int(rows[0]), int(cols[0]))
                )

        observed_values = y_repr[mask]
        if observed_values.size:
            with np.errstate(invalid="ignore"):
                is_indicator = np.asarray(
                    (observed_values == 0) | (observed_values == 1),
                    dtype=bool)
            if not np.all(is_indicator):
                bad = observed_values[~is_indicator][0]
                raise ValueError(
                    "Observed labels must be 0 or 1; found %r. Mark "
                    "unaudited cells with NaN/sample_mask or pass "
                    "unknown_label." % (bad,)
                )

        return y_repr, mask

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, X, y, sample_mask=None):
        """Fit one binary classifier per class on its observed population.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Feature matrix.  Sparse CSR/CSC/COO input is supported.

        y : array-like or sparse matrix, shape (n_samples, n_classes)
            Multi-label indicator target.  Unknown cells are NaN,
            ``unknown_label``, or cells where ``sample_mask`` is False.

        sample_mask : array-like or sparse matrix, shape (n_samples, n_classes)
            Observation mask; True means the label was audited.  If None,
            all non-unknown target cells are considered observed.

        Returns
        -------
        self : MultiLabelClassifier
        """
        self._validate_params()

        X = check_array(X, accept_sparse=("csr", "csc", "coo"),
                        dtype=np.float64, ensure_2d=True, allow_nd=False)
        n_samples, n_features = X.shape

        target = self._resolve_classes(y)
        self.classes_ = np.array(target, copy=True)
        n_classes = len(self.classes_)
        self.n_classes_ = n_classes
        self.n_features_in_ = n_features

        y_repr, mask = self._prepare_targets(
            y, sample_mask, n_samples, self.classes_)

        base_estimator = self._make_base_estimator()
        estimators = []
        model_kind = np.empty(n_classes, dtype=object)
        n_observed = np.zeros(n_classes, dtype=np.int64)
        n_positives = np.zeros(n_classes, dtype=np.int64)
        class_priors = np.full(n_classes, np.nan, dtype=np.float64)

        for k in range(n_classes):
            mask_k = mask[:, k]
            nk = int(mask_k.sum())
            n_observed[k] = nk
            class_label = self.classes_[k]

            if nk == 0:
                if self.on_unobserved == "raise":
                    raise ValueError(
                        "Class %r (column %d) has no observed labels. "
                        "Use on_unobserved='prior' to fall back to a "
                        "constant prior model."
                        % (class_label, k)
                    )
                model = _ClassModel(
                    None, "unobserved_prior", n_features,
                    n_observed=nk, n_positive=0,
                    prior=float(self.unobserved_prior))

            else:
                column = _column_values(y_repr, n_samples, k)
                values_k = np.asarray(column)[mask_k]
                npos = int(np.sum(values_k == 1))
                n_positives[k] = npos
                class_priors[k] = npos / nk

                if npos == 0 or npos == nk:
                    if self.on_single_class == "raise":
                        raise ValueError(
                            "Class %r (column %d) has only one observed "
                            "label value (%d observed, %d positive). Use "
                            "on_single_class='prior' to fall back to a "
                            "constant prior model."
                            % (class_label, k, nk, npos)
                        )
                    model = _ClassModel(
                        None, "single_class_prior", n_features,
                        n_observed=nk, n_positive=npos,
                        prior=1.0 if npos == nk else 0.0)

                else:
                    estimator = clone(base_estimator)
                    estimator.fit(_subset_rows(X, mask_k),
                                  values_k.astype(np.int32))
                    self._validate_fitted_estimator(
                        estimator, n_features, class_label)
                    model = _ClassModel(
                        estimator, "fitted", n_features,
                        n_observed=nk, n_positive=npos)

            estimators.append(model)
            model_kind[k] = model.kind_

        self.estimators_ = estimators
        self.model_kind_ = model_kind
        self.n_observed_ = n_observed
        self.n_positives_ = n_positives
        self.class_priors_ = class_priors
        self.coef_ = np.vstack([model.coef_ for model in estimators])
        self.intercept_ = np.concatenate(
            [model.intercept_ for model in estimators])

        return self

    def _check_X(self, X):
        check_is_fitted(self, ["classes_", "estimators_"])
        X = check_array(X, accept_sparse=("csr", "csc", "coo"),
                        dtype=np.float64, ensure_2d=True, allow_nd=False)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has %d features, but MultiLabelClassifier was fitted "
                "with %d features."
                % (X.shape[1], self.n_features_in_)
            )
        return X

    def decision_function(self, X):
        """Decision scores for the full class axis.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray, shape (n_samples, n_classes)
        """
        X = self._check_X(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_

    def predict_proba(self, X):
        """Positive-class probability for every class on the full axis.

        Uses the per-class estimator's native ``predict_proba`` when it is
        available and falls back to the logistic transform of the decision
        score otherwise.  Constant fallback classes return their prior.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
        """
        X = self._check_X(X)
        proba = np.empty((X.shape[0], self.n_classes_), dtype=np.float64)
        for k, model in enumerate(self.estimators_):
            proba[:, k] = model.predict_proba(X)[:, 1]
        return proba

    def predict(self, X):
        """Predict 0/1 indicators for the full class axis.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)

        Returns
        -------
        predictions : ndarray, shape (n_samples, n_classes)
        """
        X = self._check_X(X)
        predictions = np.empty((X.shape[0], self.n_classes_), dtype=np.int32)
        for k, model in enumerate(self.estimators_):
            predictions[:, k] = model.predict(X, threshold=self.threshold)
        return predictions

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def _eval_arrays(self, X, y, sample_mask):
        X = self._check_X(X)
        n_samples = X.shape[0]
        y_repr, mask = self._prepare_targets(
            y, sample_mask, n_samples, self.classes_)

        truth = np.zeros(mask.shape, dtype=np.float64)
        for k in range(self.n_classes_):
            column = _column_values(y_repr, n_samples, k)
            truth[mask[:, k], k] = np.asarray(column)[mask[:, k]]

        pred = self.predict(X)
        return pred, truth, mask

    def per_class_score(self, X, y, sample_mask=None):
        """Mask-aware metrics, computed separately for every class.

        Only audited cells (``sample_mask`` True) are counted.  Rows are
        never dropped wholesale because of one unknown cell.  Classes with
        no audited cell get NaN metrics.

        Returns
        -------
        report : dict
            Keys ``classes``, ``n_observed``, ``n_positive``, ``accuracy``,
            ``precision``, ``recall`` and ``f1`` (arrays of length
            ``n_classes``).  Undefined precision/recall/F1 for an observed
            class are reported as 0.0; metrics for unobserved classes are
            NaN.
        """
        pred, truth, mask = self._eval_arrays(X, y, sample_mask)

        observed = mask.sum(axis=0)
        positives = (mask & (truth == 1)).sum(axis=0)
        tp = (mask & (pred == 1) & (truth == 1)).sum(axis=0)
        fp = (mask & (pred == 1) & (truth == 0)).sum(axis=0)
        fn = (mask & (pred == 0) & (truth == 1)).sum(axis=0)
        correct = (mask & (pred == truth)).sum(axis=0)

        n_classes = self.n_classes_
        accuracy = np.full(n_classes, np.nan)
        precision = np.full(n_classes, np.nan)
        recall = np.full(n_classes, np.nan)
        f1 = np.full(n_classes, np.nan)

        seen = observed > 0
        accuracy[seen] = correct[seen] / observed[seen]

        denom = tp + fp
        precision[seen] = np.where(
            denom[seen] > 0, tp[seen] / np.maximum(denom[seen], 1), 0.0)

        denom = tp + fn
        recall[seen] = np.where(
            denom[seen] > 0, tp[seen] / np.maximum(denom[seen], 1), 0.0)

        denom = 2 * tp + fp + fn
        f1[seen] = np.where(
            denom[seen] > 0, (2 * tp[seen]) / np.maximum(denom[seen], 1), 0.0)

        return {
            "classes": self.classes_,
            "n_observed": observed.astype(np.int64),
            "n_positive": positives.astype(np.int64),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def score(self, X, y, sample_mask=None, scoring="accuracy"):
        """Mask-aware aggregate score over audited cells.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)

        y : array-like or sparse matrix, shape (n_samples, n_classes)

        sample_mask : array-like or sparse matrix or None
            True means audited.  Only audited cells contribute to the score;
            unaudited cells are skipped, and entire rows are not dropped.

        scoring : {'accuracy', 'precision', 'recall', 'f1'}, default='accuracy'
            Micro-averaged metric over all audited cells.

        Returns
        -------
        score : float
        """
        if scoring not in ("accuracy", "precision", "recall", "f1"):
            raise ValueError(
                "scoring must be one of 'accuracy', 'precision', 'recall' "
                "or 'f1', got %r." % (scoring,)
            )

        pred, truth, mask = self._eval_arrays(X, y, sample_mask)
        n_observed = int(mask.sum())
        if n_observed == 0:
            raise ValueError(
                "No observed cells to score: sample_mask selects nothing."
            )

        if scoring == "accuracy":
            return float(np.sum(pred[mask] == truth[mask]) / n_observed)

        tp = int(np.sum((pred == 1) & (truth == 1) & mask))
        fp = int(np.sum((pred == 1) & (truth == 0) & mask))
        fn = int(np.sum((pred == 0) & (truth == 1) & mask))

        if scoring == "precision":
            return float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        if scoring == "recall":
            return float(tp / (tp + fn)) if tp + fn > 0 else 0.0

        denom = 2 * tp + fp + fn
        return float(2 * tp / denom) if denom > 0 else 0.0
