# -*- coding: utf-8 -*-
import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError

from lightning.classification import MultiLabelClassifier
from lightning.classification import SGDClassifier
from lightning.classification import LinearSVC


def _make_multilabel(n_samples=240, n_features=12, n_informative=6,
                     n_classes=4, random_state=0):
    """Generate a multi-label problem with column-wise support imbalance."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    W = rng.randn(n_features, n_classes)
    scores = X.dot(W)
    Y = (scores > 0).astype(np.int32)
    return X, Y


def _random_mask(Y, fraction, random_state=0):
    rng = np.random.RandomState(random_state)
    return rng.rand(*Y.shape) < fraction


# ----------------------------------------------------------------------
# the core guarantee: unknown cells are not negatives
# ----------------------------------------------------------------------
def test_unknown_cells_are_not_negative():
    """Fitting on a masked target equals fitting on the observed cells only."""
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.6, random_state=1)
    # Give the mask an unobserved class entirely.
    M[:, -1] = False

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=np.arange(Y.shape[1]),
        on_unobserved="prior")
    clf.fit(X, Y, sample_mask=M)

    for k in range(Y.shape[1]):
        mask_k = M[:, k]
        nk = int(mask_k.sum())
        assert clf.n_observed_[k] == nk

        if nk == 0:
            # Never audited: explicit prior fallback, no fitted model.
            assert clf.model_kind_[k] == "unobserved_prior"
            assert clf.estimators_[k].estimator_ is None
            np.testing.assert_array_equal(clf.estimators_[k].coef_, 0.0)
            continue

        kind = clf.model_kind_[k]
        model = clf.estimators_[k]

        if kind != "fitted":
            assert model.estimator_ is None
            continue

        # Reproduce the per-class training population by hand: only the
        # observed rows, and only with their audited targets.
        reference = SGDClassifier(loss="log", random_state=0)
        reference.fit(X[mask_k], Y[mask_k, k])

        np.testing.assert_allclose(clf.coef_[k], reference.coef_[0],
                                   rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(
            clf.predict_proba(X)[:, k],
            reference.predict_proba(X)[:, 1], rtol=1e-8, atol=1e-8)

        # The negative count used by the training layer is the observed
        # negative count, not "all rows minus positives".
        observed_negatives = int(np.sum(Y[mask_k, k] == 0))
        assert model.n_observed_ - model.n_positive_ == observed_negatives


def test_unknown_not_encoded_as_zero_changes_the_model():
    """Treating unknown as negative gives a different coefficient vector."""
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.5, random_state=2)

    clf_mask = MultiLabelClassifier(
        base_estimator=LinearSVC(C=1.0, random_state=0),
        classes=np.arange(Y.shape[1]))
    clf_mask.fit(X, Y, sample_mask=M)

    clf_naive = MultiLabelClassifier(
        base_estimator=LinearSVC(C=1.0, random_state=0),
        classes=np.arange(Y.shape[1]))
    # Unknown cells all encoded as 0 == negative (the buggy behaviour).
    clf_naive.fit(X, np.where(M, Y, 0))

    k = 0
    assert clf_mask.model_kind_[k] == "fitted"
    assert clf_naive.n_observed_[k] == X.shape[0]
    assert clf_mask.n_observed_[k] < X.shape[0]

    assert not np.allclose(clf_mask.coef_[k], clf_naive.coef_[k])


def test_mask_is_not_a_feature():
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.5, random_state=3)

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=np.arange(Y.shape[1]))
    clf.fit(X, Y, sample_mask=M)

    assert clf.n_features_in_ == X.shape[1]
    assert clf.coef_.shape == (Y.shape[1], X.shape[1])
    assert clf.intercept_.shape == (Y.shape[1],)

    # Decision scores do not depend on the mask of the evaluation batch:
    # there is no mask argument at prediction time at all.
    probe = np.vstack([np.zeros((1, X.shape[1])), X[:5]])
    scores = clf.decision_function(probe)
    assert scores.shape == (6, Y.shape[1])
    np.testing.assert_allclose(scores[0], clf.intercept_)


def test_mask_does_not_change_prediction_api():
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.5)

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=np.arange(Y.shape[1]))
    clf.fit(X, Y, sample_mask=M)

    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    scores = clf.decision_function(X)
    assert pred.shape == proba.shape == scores.shape == Y.shape
    assert set(np.unique(pred)) <= {0, 1}
    np.testing.assert_array_equal(
        pred, (proba > clf.threshold).astype(np.int32))


# ----------------------------------------------------------------------
# class vocabulary / axis
# ----------------------------------------------------------------------
def test_full_class_axis_with_unobserved_class():
    X, Y = _make_multilabel(n_classes=4)
    M = np.ones_like(Y, dtype=bool)
    M[:, 2] = False

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=np.array(["a", "b", "c", "d"]),
        unobserved_prior=0.3)
    clf.fit(X, Y, sample_mask=M)

    np.testing.assert_array_equal(clf.classes_, ["a", "b", "c", "d"])
    assert clf.model_kind_[2] == "unobserved_prior"
    assert clf.estimators_[2].prior_ == pytest.approx(0.3)

    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 4)
    np.testing.assert_allclose(proba[:, 2], 0.3)
    assert clf.predict(X).shape == (X.shape[0], 4)

    # The unobserved class still has a slot in the aggregated matrix.
    assert clf.coef_.shape == (4, X.shape[1])
    np.testing.assert_array_equal(clf.coef_[2], 0.0)


def test_frozen_vocabulary_rejects_new_columns():
    X, Y = _make_multilabel(n_classes=3)
    clf = MultiLabelClassifier(classes=["a", "b", "c"])
    clf.fit(X, Y)
    assert clf.model_kind_.tolist() == ["fitted"] * 3

    # Different number of columns: rejected.
    with pytest.raises(ValueError, match="frozen"):
        clf.fit(X, Y[:, :2])

    # An explicit vocabulary swapped between fits: rejected.
    clf2 = MultiLabelClassifier(classes=["a", "b"])
    clf2.fit(X, Y[:, :2])
    clf2.classes = ["b", "a"]
    with pytest.raises(ValueError, match="frozen"):
        clf2.fit(X, Y[:, :2])

    # A new estimator rejects classes that do not match the target width.
    with pytest.raises(ValueError, match="classes has"):
        MultiLabelClassifier(classes=["a", "b"]).fit(X, Y)


def test_classes_frozen_after_first_fit():
    X, Y = _make_multilabel(n_classes=3)
    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0))
    clf.fit(X, Y)
    np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    # Re-fitting without explicit classes keeps the frozen vocabulary.
    clf.fit(X, Y)
    np.testing.assert_array_equal(clf.classes_, [0, 1, 2])


# ----------------------------------------------------------------------
# sparse X and sparse y
# ----------------------------------------------------------------------
def test_sparse_X_support():
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.6)
    X_sparse = sp.csr_matrix(X)

    clf_dense = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(Y.shape[1]))
    clf_dense.fit(X, Y, sample_mask=M)

    clf_sparse = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(Y.shape[1]))
    clf_sparse.fit(X_sparse, Y, sample_mask=M)

    np.testing.assert_allclose(clf_sparse.coef_, clf_dense.coef_,
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        clf_sparse.decision_function(X_sparse),
        clf_dense.decision_function(X), rtol=1e-10, atol=1e-12)


def test_sparse_y_and_sparse_mask():
    X, Y = _make_multilabel()
    M = _random_mask(Y, 0.7)
    Y_sparse = sp.csr_matrix(Y)
    M_sparse = sp.csr_matrix(M.astype(np.int8))

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(Y.shape[1]))
    clf.fit(X, Y_sparse, sample_mask=M_sparse)

    reference = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(Y.shape[1]))
    reference.fit(X, Y, sample_mask=M)

    np.testing.assert_array_equal(clf.n_observed_, reference.n_observed_)
    np.testing.assert_allclose(clf.coef_, reference.coef_)


# ----------------------------------------------------------------------
# different per-class support
# ----------------------------------------------------------------------
def test_different_class_support():
    X, Y = _make_multilabel(n_classes=4, random_state=5)
    M = np.zeros_like(Y, dtype=bool)
    M[:200, 0] = True            # plenty of support
    M[:40, 1] = True             # little support
    M[:10, 2] = True             # tiny support
    M[:, 3] = False              # no support at all

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=np.arange(4))
    clf.fit(X, Y, sample_mask=M)

    np.testing.assert_array_equal(clf.n_observed_, [200, 40, 10, 0])
    assert clf.model_kind_[3] == "unobserved_prior"
    # A small-support class still gets its own model on its own population.
    assert clf.n_observed_[2] == 10
    if clf.model_kind_[2] == "fitted":
        reference = SGDClassifier(loss="log", random_state=0)
        reference.fit(X[:10], Y[:10, 2])
        np.testing.assert_allclose(clf.coef_[2], reference.coef_[0])


# ----------------------------------------------------------------------
# fallbacks and rejection policies
# ----------------------------------------------------------------------
def test_all_unknown_class_prior_and_raise():
    X, Y = _make_multilabel(n_classes=3)
    M = np.ones_like(Y, dtype=bool)
    M[:, 1] = False

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(3), on_unobserved="prior",
        unobserved_prior=0.25)
    clf.fit(X, Y, sample_mask=M)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], 0.25)
    assert clf.model_kind_[1] == "unobserved_prior"

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(3), on_unobserved="raise")
    with pytest.raises(ValueError, match="no observed labels"):
        clf.fit(X, Y, sample_mask=M)


def test_single_class_prior_and_raise():
    X, Y = _make_multilabel(n_classes=3)
    M = np.ones_like(Y, dtype=bool)

    # Class 0 all positive among observed rows.
    Y = Y.copy()
    Y[M[:, 0], 0] = 1
    # Class 1 all negative among observed rows.
    Y[M[:, 1], 1] = 0

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(3), on_single_class="prior")
    clf.fit(X, Y, sample_mask=M)
    assert clf.model_kind_[0] == "single_class_prior"
    assert clf.model_kind_[1] == "single_class_prior"
    assert clf.model_kind_[2] == "fitted"
    # The clipped prior never becomes exactly 0 or 1.
    p0 = clf.predict_proba(X)[:, 0]
    p1 = clf.predict_proba(X)[:, 1]
    assert np.all(p0 > 0.99) and np.all(p0 < 1.0)
    assert np.all(p1 < 0.01) and np.all(p1 > 0.0)

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=np.arange(3), on_single_class="raise")
    with pytest.raises(ValueError, match="only one observed label value"):
        clf.fit(X, Y, sample_mask=M)


def test_invalid_policy_rejected():
    X, Y = _make_multilabel(n_classes=2)
    with pytest.raises(ValueError, match="on_unobserved"):
        MultiLabelClassifier(on_unobserved="ignore").fit(X, Y)
    with pytest.raises(ValueError, match="on_single_class"):
        MultiLabelClassifier(on_single_class="ignore").fit(X, Y)


# ----------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------
def test_observed_non_binary_rejected():
    X, Y = _make_multilabel(n_classes=2)
    Y = Y.copy()
    Y[0, 0] = 2
    with pytest.raises(ValueError, match="Observed labels must be 0 or 1"):
        MultiLabelClassifier(classes=[0, 1]).fit(X, Y)


def test_mask_sentinel_conflict_rejected():
    X, Y = _make_multilabel(n_classes=2)
    M = np.ones_like(Y, dtype=bool)
    Yf = Y.astype(np.float64)
    Yf[3, 0] = np.nan
    with pytest.raises(ValueError, match="marked observed"):
        MultiLabelClassifier(classes=[0, 1]).fit(X, Yf, sample_mask=M)


def test_nan_marks_unknown():
    X, Y = _make_multilabel(n_classes=3)
    Yf = Y.astype(np.float64)
    M = np.ones_like(Y, dtype=bool)
    M[:, 2] = False
    Yf[~M] = np.nan

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2])
    clf.fit(X, Yf)               # no explicit mask: NaN is unknown
    np.testing.assert_array_equal(clf.n_observed_, M.sum(axis=0))
    assert clf.model_kind_[2] == "unobserved_prior"

    # Explicit mask + NaN is even allowed as long as they do not conflict.
    clf.fit(X, Yf, sample_mask=M)


def test_unknown_label_sentinel():
    X, Y = _make_multilabel(n_classes=3)
    M = np.ones_like(Y, dtype=bool)
    M[:, 1] = False
    Ys = Y.copy()
    Ys[~M] = -1

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2],
        unknown_label=-1)
    clf.fit(X, Ys)
    np.testing.assert_array_equal(clf.n_observed_, M.sum(axis=0))
    assert clf.model_kind_[1] == "unobserved_prior"

    # A sentinel on a cell marked observed is rejected.
    with pytest.raises(ValueError, match="marked observed"):
        clf2 = MultiLabelClassifier(classes=[0, 1, 2], unknown_label=-1)
        clash = M.copy()
        clash[0, 1] = True
        clf2.fit(X, Ys, sample_mask=clash)


def test_1d_target_rejected():
    X, Y = _make_multilabel(n_classes=2)
    with pytest.raises(ValueError, match="2-dimensional"):
        MultiLabelClassifier(classes=[0, 1]).fit(X, Y[:, 0])
    clf = MultiLabelClassifier(classes=[0, 1]).fit(X, Y)
    with pytest.raises(ValueError, match="2-dimensional"):
        clf.score(X, Y[:, 0])


def test_mask_shape_mismatch_rejected():
    X, Y = _make_multilabel(n_classes=2)
    with pytest.raises(ValueError, match="sample_mask"):
        MultiLabelClassifier(classes=[0, 1]).fit(
            X, Y, sample_mask=np.ones((X.shape[0], 3), dtype=bool))


def test_not_fitted_raises():
    X, _ = _make_multilabel(n_classes=2)
    with pytest.raises(NotFittedError):
        MultiLabelClassifier().predict(X)


def test_kernel_estimator_rejected():
    X, Y = _make_multilabel(n_classes=2)
    from lightning.classification import KernelSVC
    clf = MultiLabelClassifier(
        base_estimator=KernelSVC(kernel="linear"), classes=[0, 1])
    with pytest.raises(ValueError, match="linear binary"):
        with pytest.warns(DeprecationWarning):
            clf.fit(X, Y)


def test_default_base_estimator():
    X, Y = _make_multilabel(n_classes=3)
    clf = MultiLabelClassifier(classes=[0, 1, 2]).fit(X, Y)
    assert isinstance(clf.estimators_[0].estimator_, SGDClassifier)
    assert clf.predict(X).shape == Y.shape


# ----------------------------------------------------------------------
# evaluation respects the mask
# ----------------------------------------------------------------------
def test_score_uses_only_observed_cells():
    X, Y = _make_multilabel(n_classes=3)
    M = _random_mask(Y, 0.5, random_state=7)
    M[:, 2] = False

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2])
    clf.fit(X, Y, sample_mask=M)

    # Deliberately corrupt the unobserved class; the score must not notice.
    Y_wrong = Y.copy()
    Y_wrong[:, 2] = 1 - Y_wrong[:, 2]
    score_a = clf.score(X, Y, sample_mask=M)
    score_b = clf.score(X, Y_wrong, sample_mask=M)
    assert score_a == pytest.approx(score_b)

    # Across audited cells only.
    pred = clf.predict(X)
    expected = np.sum(pred[M] == Y[M]) / M.sum()
    assert score_a == pytest.approx(expected)

    # A row is never dropped wholesale, even with several unknown cells.
    partial = np.zeros_like(M)
    partial[5, 0] = True
    assert clf.score(X, Y, sample_mask=partial) == pytest.approx(
        float(pred[5, 0] == Y[5, 0]))


def test_per_class_score_reports_observed_support():
    X, Y = _make_multilabel(n_classes=3)
    M = np.zeros_like(Y, dtype=bool)
    M[:100, 0] = True
    M[:50, 1] = True
    # class 2 never observed

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2])
    clf.fit(X, Y, sample_mask=M)
    report = clf.per_class_score(X, Y, sample_mask=M)

    np.testing.assert_array_equal(report["n_observed"], [100, 50, 0])
    assert np.isnan(report["accuracy"][2])
    assert not np.isnan(report["accuracy"][0])
    assert 0.0 <= report["f1"][0] <= 1.0


def test_score_rejects_empty_mask():
    X, Y = _make_multilabel(n_classes=2)
    clf = MultiLabelClassifier(classes=[0, 1]).fit(X, Y)
    with pytest.raises(ValueError, match="No observed cells"):
        clf.score(X, Y, sample_mask=np.zeros_like(Y, dtype=bool))


def test_score_unknown_label_roundtrip():
    X, Y = _make_multilabel(n_classes=3)
    M = _random_mask(Y, 0.5, random_state=11)
    Ys = Y.astype(np.float64)
    Ys[~M] = np.nan

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2])
    clf.fit(X, Ys)
    assert clf.score(X, Ys) == pytest.approx(
        clf.score(X, Y, sample_mask=M))


# ----------------------------------------------------------------------
# per-class field contract
# ----------------------------------------------------------------------
def test_uniform_field_contract_for_every_class():
    X, Y = _make_multilabel(n_classes=4)
    M = np.ones_like(Y, dtype=bool)
    M[:, 3] = False
    Y[:, 2] = 0                    # single-class support

    clf = MultiLabelClassifier(
        base_estimator=SGDClassifier(loss="log", random_state=0),
        classes=[0, 1, 2, 3])
    clf.fit(X, Y, sample_mask=M)

    assert set(clf.model_kind_) == {
        "fitted", "single_class_prior", "unobserved_prior"}

    for k, model in enumerate(clf.estimators_):
        assert np.array_equal(model.classes_, [0, 1])
        assert model.coef_.shape == (1, X.shape[1])
        assert model.intercept_.shape == (1,)
        scores = model.decision_function(X)
        proba = model.predict_proba(X)
        pred = model.predict(X)
        assert scores.shape == (X.shape[0],)
        assert proba.shape == (X.shape[0], 2)
        assert pred.shape == (X.shape[0],)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        # The wrapper's fields agree with the aggregated matrix.
        np.testing.assert_allclose(scores, clf.decision_function(X)[:, k])
        np.testing.assert_allclose(proba[:, 1], clf.predict_proba(X)[:, k])


def test_decision_function_and_proba_agree_for_fitted_class():
    X, Y = _make_multilabel(n_classes=2)
    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1])
    clf.fit(X, Y)

    scores = clf.decision_function(X)
    proba = clf.predict_proba(X)
    # LinearSVC has no predict_proba: the logistic transform is used.
    np.testing.assert_allclose(proba, 1.0 / (1.0 + np.exp(-scores)))


def test_pickle_roundtrip():
    import pickle
    X, Y = _make_multilabel(n_classes=3)
    M = _random_mask(Y, 0.6)
    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0), classes=[0, 1, 2])
    clf.fit(X, Y, sample_mask=M)

    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.predict_proba(X),
                               clf.predict_proba(X))


def test_dataframe_columns_align():
    pd = pytest.importorskip("pandas")
    X, Y = _make_multilabel(n_classes=3)
    Y_df = pd.DataFrame(Y, columns=["sports", "tech", "politics"])

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0))
    clf.fit(X, Y_df)
    np.testing.assert_array_equal(
        clf.classes_, ["sports", "tech", "politics"])

    # Columns presented in a different order are realigned, not misread.
    reference = clf.predict_proba(X)
    shuffled = Y_df[["politics", "sports", "tech"]]
    clf.fit(X, shuffled)
    np.testing.assert_array_equal(clf.classes_,
                                  ["sports", "tech", "politics"])
    np.testing.assert_allclose(clf.predict_proba(X), reference)

    # Wrong vocabulary is rejected.
    with pytest.raises(ValueError, match="do not match"):
        clf.fit(X, Y_df[["sports", "tech", "other"]])


def test_multiclass_target_semantics():
    """A one-hot target is treated as independent binary columns."""
    X, y = make_classification(n_samples=150, n_features=8, n_informative=5,
                               n_classes=3, random_state=0)
    Y = np.zeros((len(y), 3), dtype=np.int32)
    Y[np.arange(len(y)), y] = 1

    clf = MultiLabelClassifier(
        base_estimator=LinearSVC(random_state=0),
        classes=[0, 1, 2])
    clf.fit(X, Y)
    pred = clf.predict(X)
    assert pred.shape == Y.shape
    # Independent binary columns: no cross-column normalisation is applied,
    # so a row could in principle be positive for several classes.  Here we
    # only check that the true class column is recovered far above the 1/3
    # base rate rather than applying a multiclass argmax.
    true_column_rate = np.mean(pred[np.arange(len(y)), y])
    assert true_column_rate > 0.7
