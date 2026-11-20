import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.datasets import make_regression

from lightning.impl.datasets.samples_generator import make_classification
from lightning.impl.datasets.samples_generator import make_nn_regression
from lightning.impl.sgd import SGDClassifier
from lightning.impl.sgd import SGDRegressor
from lightning.impl.tests.utils import check_predict_proba


@pytest.fixture(scope="module")
def reg_train_data():
    X, y = make_regression(n_samples=100, n_features=10, n_informative=8,
                           random_state=0)
    return X, y


@pytest.fixture(scope="module")
def reg_nn_train_data():
    X, y, _ = make_nn_regression(n_samples=100, n_features=10, n_informative=8,
                                 random_state=0)
    return X, y


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
@pytest.mark.parametrize("clf", [SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="pegasos"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=False, learning_rate="pegasos"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="invscaling"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="squared_hinge",
                                               eta0=1e-2,
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="log",
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="modified_huber",
                                               fit_intercept=True, learning_rate="constant")])
def test_binary_linear_sgd(data, clf, request):
    X, y = request.getfixturevalue(data)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.934
    assert list(clf.classes_) == [0, 1]
    if clf.loss in {'log', 'modified_huber'}:
        check_predict_proba(clf, X)
    else:
        assert not hasattr(clf, 'predict_proba')


def test_multiclass_sgd(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = SGDClassifier(random_state=0)
    clf.fit(mult_dense, mult_target)
    assert clf.score(mult_dense, mult_target) > 0.80
    assert list(clf.classes_) == [0, 1, 2]


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_hinge_sgd(data, fit_intercept, request):
    X, y = request.getfixturevalue(data)
    clf = SGDClassifier(loss="hinge", multiclass=True,
                        fit_intercept=fit_intercept, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_multiclass_hinge_sgd_l1l2(data, request):
    X, y = request.getfixturevalue(data)
    clf = SGDClassifier(loss="hinge", penalty="l1/l2",
                        multiclass=True, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.75


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_squared_hinge_sgd(data, fit_intercept, request):
    X, y = request.getfixturevalue(data)
    clf = SGDClassifier(loss="squared_hinge", multiclass=True,
                        learning_rate="constant", eta0=1e-3,
                        fit_intercept=fit_intercept, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_log_sgd(data, fit_intercept, request):
    X, y = request.getfixturevalue(data)
    clf = SGDClassifier(loss="log", multiclass=True,
                        fit_intercept=fit_intercept,
                        random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


def test_regression_squared_loss(reg_train_data):
    X, y = reg_train_data
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 4.749, 3)


@pytest.mark.parametrize("alpha", [0, 1e-6])
def test_regression_squared_loss_nn_l1(reg_nn_train_data, alpha):
    X, y = reg_nn_train_data
    reg = SGDRegressor(loss="squared", penalty="nn", learning_rate="constant",
                       eta0=1e-1, alpha=alpha, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 0.016, 3)
    assert (reg.coef_ >= 0).all()


def test_regression_squared_loss_nn_l2(reg_nn_train_data):
    X, y = reg_nn_train_data

    reg = SGDRegressor(loss="squared", penalty="nnl2", learning_rate="constant",
                       eta0=1e-1, alpha=1e-4, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 0.016, 3)
    np.testing.assert_almost_equal(reg.coef_.sum(), 2.131, 3)
    assert (reg.coef_ >= 0).all()


def test_regression_squared_loss_multiple_output(reg_train_data):
    X, y = reg_train_data
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0, max_iter=10)
    Y = np.zeros((len(y), 2))
    Y[:, 0] = y
    Y[:, 1] = y
    reg.fit(X, Y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - Y) ** 2), 4.397, 3)


# ---------------------------------------------------------------------------
# partial_fit
# ---------------------------------------------------------------------------

def _batches(X, y, n_batches):
    edges = np.linspace(0, X.shape[0], n_batches + 1).astype(int)
    for i in range(n_batches):
        sl = slice(edges[i], edges[i + 1])
        yield X[sl], y[sl]


@pytest.mark.parametrize("penalty", ["l2", "l1"])
@pytest.mark.parametrize("learning_rate",
                         ["constant", "pegasos", "invscaling"])
def test_partial_fit_binary_matches_single_pass(penalty, learning_rate):
    """Fixed order, no shuffle: streamed updates equal one ordered pass."""
    X, y = bin_dense = make_classification(n_samples=200, n_features=40,
                                           n_informative=5, n_classes=2,
                                           random_state=0)
    kwargs = dict(random_state=0, shuffle=False, max_iter=1,
                  loss="hinge", penalty=penalty, learning_rate=learning_rate,
                  eta0=0.01)
    clf_full = SGDClassifier(**kwargs).fit(X, y)

    clf_inc = SGDClassifier(**kwargs)
    for Xb, yb in _batches(X, y, 5):
        clf_inc.partial_fit(Xb, yb, classes=np.unique(y))

    np.testing.assert_allclose(clf_full.coef_, clf_inc.coef_, rtol=1e-10)
    np.testing.assert_allclose(clf_full.intercept_, clf_inc.intercept_,
                               rtol=1e-10)
    # the learning-rate step counter counts every update performed
    assert clf_inc._sgd_t_ == X.shape[0]


@pytest.mark.parametrize("multiclass", [False, True])
def test_partial_fit_multiclass_matches_single_pass(multiclass):
    X, y = make_classification(n_samples=300, n_features=40,
                               n_informative=5, n_classes=3, random_state=1)
    kwargs = dict(random_state=0, shuffle=False, max_iter=1,
                  multiclass=multiclass, learning_rate="constant",
                  eta0=0.01)

    clf_one = SGDClassifier(**kwargs)
    clf_one.partial_fit(X, y, classes=np.unique(y))

    clf_inc = SGDClassifier(**kwargs)
    for Xb, yb in _batches(X, y, 6):
        clf_inc.partial_fit(Xb, yb, classes=np.unique(y))

    np.testing.assert_allclose(clf_one.coef_, clf_inc.coef_, rtol=1e-10)
    np.testing.assert_allclose(clf_one.intercept_, clf_inc.intercept_,
                               rtol=1e-10)
    assert clf_inc._sgd_t_ == X.shape[0]


def test_partial_fit_sparse_matches_dense():
    """Sparse batches take the sparse code path and give identical weights."""
    X, y = make_classification(n_samples=200, n_features=40, n_informative=5,
                               n_classes=2, random_state=2)
    Xs = sp.csr_matrix(X)
    kwargs = dict(random_state=0, shuffle=False, max_iter=1,
                  learning_rate="constant", eta0=0.01)

    clf_dense = SGDClassifier(**kwargs)
    clf_sparse = SGDClassifier(**kwargs)
    for (Xdb, yb), (Xsb, _) in zip(_batches(X, y, 5), _batches(Xs, y, 5)):
        assert sp.issparse(Xsb)
        clf_dense.partial_fit(Xdb, yb, classes=np.unique(y))
        clf_sparse.partial_fit(Xsb, yb, classes=np.unique(y))

    np.testing.assert_allclose(clf_sparse.coef_, clf_dense.coef_,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(clf_sparse.intercept_, clf_dense.intercept_,
                               rtol=1e-12, atol=1e-12)


def test_partial_fit_absent_class_keeps_slot():
    X, y = make_classification(n_samples=300, n_features=40,
                               n_informative=5, n_classes=3, random_state=3)
    classes = np.unique(y)

    clf = SGDClassifier(random_state=0, shuffle=False, max_iter=1,
                        learning_rate="constant", eta0=0.01)
    # first batch contains classes 0 and 1 only
    mask = y != 2
    clf.partial_fit(X[mask], y[mask], classes=classes)

    assert list(clf.classes_) == [0, 1, 2]
    assert clf.coef_.shape[0] == 3
    # the absent class keeps its slot and an untouched zero vector
    assert np.all(clf.coef_[2] == 0.0)

    # later batches containing class 2 update that same slot
    clf.partial_fit(X[y == 2][:20], np.full(20, 2))
    assert list(clf.classes_) == [0, 1, 2]
    assert np.any(clf.coef_[2] != 0.0)


def test_partial_fit_binary_absent_label():
    X, y = make_classification(n_samples=200, n_features=40, n_informative=5,
                               n_classes=2, random_state=4)
    clf = SGDClassifier(random_state=0, shuffle=False, max_iter=1,
                        learning_rate="constant", eta0=0.01)
    clf.partial_fit(X[y == 0][:30], np.zeros(30, dtype=np.int32),
                    classes=np.array([0, 1]))
    assert list(clf.classes_) == [0, 1]
    assert clf.coef_.shape[0] == 1
    clf.partial_fit(X[y == 1][:30], np.ones(30, dtype=np.int32))
    assert set(np.unique(clf.predict(X))).issubset({0, 1})


def test_partial_fit_rejects_unknown_labels():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=5,
                               n_classes=2, random_state=5)
    clf = SGDClassifier(random_state=0)
    clf.partial_fit(X[:50], y[:50], classes=np.array([0, 1]))

    with pytest.raises(ValueError, match="unknown label"):
        clf.partial_fit(X[:10], np.full(10, 7, dtype=np.int32))

    # direct multiclass formulation rejects unknown labels too
    clf_mc = SGDClassifier(multiclass=True, random_state=0)
    clf_mc.partial_fit(X[:50], y[:50], classes=np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="unknown label"):
        clf_mc.partial_fit(X[:5], np.full(5, 3, dtype=np.int32))


def test_partial_fit_requires_classes_on_first_call():
    X, y = make_classification(n_samples=50, n_features=10, n_classes=2,
                               random_state=6)
    clf = SGDClassifier(random_state=0)
    with pytest.raises(ValueError, match="classes must be passed"):
        clf.partial_fit(X, y)


def test_partial_fit_rejects_feature_mismatch():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                               n_classes=2, random_state=7)
    clf = SGDClassifier(random_state=0)
    clf.partial_fit(X[:50], y[:50], classes=np.unique(y))
    with pytest.raises(ValueError, match="features"):
        clf.partial_fit(X[:10, :10], y[:10])


def test_partial_fit_preserves_rng_and_schedule_state():
    X, y = make_classification(n_samples=200, n_features=30, n_informative=5,
                               n_classes=2, random_state=8)

    def streamed(seed):
        clf = SGDClassifier(random_state=seed, shuffle=True, max_iter=1,
                            learning_rate="constant", eta0=0.01)
        for Xb, yb in _batches(X, y, 5):
            clf.partial_fit(Xb, yb, classes=np.unique(y))
        return clf

    np.testing.assert_array_equal(streamed(0).coef_, streamed(0).coef_)

    # pegasos: the step counter (hence the eta schedule) keeps advancing
    clf = SGDClassifier(random_state=0, shuffle=False, max_iter=1,
                        learning_rate="pegasos")
    clf.partial_fit(X[:60], y[:60], classes=np.unique(y))
    assert clf._sgd_t_ == 60
    clf.partial_fit(X[60:120], y[60:120])
    assert clf._sgd_t_ == 120


def test_partial_fit_state_recoverable_with_pickle():
    X, y = make_classification(n_samples=200, n_features=30, n_informative=5,
                               n_classes=2, random_state=9)
    batches = list(_batches(X, y, 4))

    clf = SGDClassifier(random_state=3, shuffle=True, max_iter=1,
                        learning_rate="constant", eta0=0.01)
    for Xb, yb in batches[:2]:
        clf.partial_fit(Xb, yb, classes=np.unique(y))

    resumed = pickle.loads(pickle.dumps(clf))
    for Xb, yb in batches[2:]:
        clf.partial_fit(Xb, yb)
        resumed.partial_fit(Xb, yb)

    np.testing.assert_array_equal(clf.coef_, resumed.coef_)
    np.testing.assert_array_equal(clf.intercept_, resumed.intercept_)
    assert clf._sgd_t_ == resumed._sgd_t_ == X.shape[0]
    np.testing.assert_array_equal(clf.predict(X), resumed.predict(X))


def test_fit_resets_partial_fit_state():
    X, y = make_classification(n_samples=200, n_features=30, n_informative=5,
                               n_classes=2, random_state=10)
    kwargs = dict(random_state=0, shuffle=False, max_iter=1,
                  learning_rate="constant", eta0=0.01)

    clf = SGDClassifier(**kwargs)
    clf.partial_fit(X[:100], y[:100], classes=np.unique(y))
    clf.fit(X, y)

    fresh = SGDClassifier(**kwargs).fit(X, y)
    np.testing.assert_array_equal(clf.coef_, fresh.coef_)
    np.testing.assert_array_equal(clf.intercept_, fresh.intercept_)
    assert clf._sgd_t_ == 0
    assert clf._sgd_random_state_ is None
    assert clf._sgd_initialized_ is False

    # a new stream starts from zero again
    clf.partial_fit(X[:50], y[:50], classes=np.unique(y))
    assert clf._sgd_t_ == 50


@pytest.mark.parametrize("penalty", ["nn", "nnl2"])
def test_partial_fit_delayed_penalty_flushed(penalty):
    """Buffered regularization must not be lost between batches."""
    X, y = make_classification(n_samples=200, n_features=30, n_informative=5,
                               n_classes=2, random_state=11)
    kwargs = dict(random_state=0, shuffle=False, max_iter=1, loss="hinge",
                  penalty=penalty, learning_rate="constant", eta0=0.01,
                  alpha=1e-4)

    clf_full = SGDClassifier(**kwargs).fit(X, y)
    clf_inc = SGDClassifier(**kwargs)
    for Xb, yb in _batches(X, y, 5):
        clf_inc.partial_fit(Xb, yb, classes=np.unique(y))

    np.testing.assert_allclose(clf_full.coef_, clf_inc.coef_, rtol=1e-10)
    assert (clf_inc.coef_ >= 0).all()
