"""
===================================
Stochastic Gradient Descent Solvers
===================================

This module provides SGD solvers for a variety of loss
functions and penalties.
"""
# Author: Mathieu Blondel
# License: BSD

import warnings

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import assert_all_finite
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import get_dataset

from .sgd_fast import _binary_sgd
from .sgd_fast import _multiclass_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive

from .sgd_fast import MulticlassLog
from .sgd_fast import MulticlassHinge
from .sgd_fast import MulticlassSquaredHinge


class _BaseSGD(object):

    def _get_penalty(self):
        penalties = {
            "nn": -1,
            "nnl1": -1,
            "nnl2": -2,
            "l1": 1,
            "l2": 2,
            "l1/l2": 12
        }
        return penalties[self.penalty]

    def _get_learning_rate(self):
        learning_rates = {"constant": 1, "pegasos": 2, "invscaling": 3}
        return learning_rates[self.learning_rate]


class SGDClassifier(BaseClassifier, _BaseSGD):
    r"""Estimator for learning linear classifiers by SGD.

    Parameters
    ----------
    loss : str, 'hinge', 'squared_hinge', 'log', 'perceptron'
        Loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    multiclass : bool
        Whether to use a direct multiclass formulation (True) or one-vs-rest
        (False). Direct formulations are only available for loss='hinge', 'squared_hinge'
        and 'log'.

    alpha : float
        Weight of the penalty term.

    learning_rate : 'pegasos', 'constant', 'invscaling'
        Learning schedule to use.

    eta0 : float
        Step size.

    power_t : float
        Power to be used (when learning_rate='invscaling').

    epsilon : float
        Value to be used for epsilon-insensitive loss.

    fit_intercept : bool
        Whether to fit the intercept or not.

    intercept_decay : float
        Value by which the intercept is multiplied (to regularize it).

    max_iter : int
        Maximum number of iterations to perform.

    shuffle : bool
        Whether to shuffle data.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.

    Notes
    -----
    Use :meth:`partial_fit` for incremental (streaming) training: the
    first call must declare the full label set with ``classes`` and
    subsequent calls continue from the preserved weights, intercept,
    learning-rate step counter and RNG state. :meth:`fit` always restarts
    training from scratch.

    Examples
    --------
    >>> from sklearn.datasets import fetch_20newsgroups_vectorized
    >>> from lightning.classification import SGDClassifier
    >>> bunch = fetch_20newsgroups_vectorized(subset="all")
    >>> X, y = bunch.data, bunch.target
    >>> clf = SGDClassifier().fit(X, y)
    >>> accuracy = clf.score(X, y)
    """

    def __init__(self, loss="hinge", penalty="l2",
                 multiclass=False, alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100, verbose=0):
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.coef_ = None
        self.intercept_ = None

    def _get_loss(self):
        if self.multiclass:
            losses = {
                "log": MulticlassLog(),
                "hinge": MulticlassHinge(),
                "squared_hinge": MulticlassSquaredHinge(),
            }
        else:
            losses = {
                "modified_huber": ModifiedHuber(),
                "hinge": Hinge(1.0),
                "squared_hinge": SquaredHinge(1.0),
                "perceptron": Hinge(0.0),
                "log": Log(),
                "squared": SquaredLoss(),
                "huber": Huber(self.epsilon),
                "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
            }
        return losses[self.loss]

    def _check_unknown_labels(self, y, known_labels=None):
        """Reject labels that were not declared on the first partial_fit."""
        if known_labels is None:
            if self.multiclass:
                known_labels = self.label_encoder_.classes_
            else:
                known_labels = self.classes_
        unknown = np.unique(y[~np.in1d(y, known_labels)])
        if unknown.size:
            raise ValueError(
                "y contains unknown label(s) %s. partial_fit only supports "
                "the classes declared on the first call: %s"
                % (list(unknown), list(np.asarray(known_labels))))

    def _init_partial_fit_labels(self, y, classes):
        """Fit label transformers on the declared, stable class set.

        The transformers are fitted on ``classes`` rather than on ``y`` so
        that a class absent from the first batch keeps its slot and labels
        are never renumbered by the classes a batch happens to contain.
        """
        classes = np.asarray(classes)
        y = np.asarray(y)

        if self.multiclass:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(classes)
            self._check_unknown_labels(y, self.label_encoder_.classes_)
            y_enc = self.label_encoder_.transform(y).astype(np.int32)
            binarizer_classes = np.arange(len(classes), dtype=np.int32)
        else:
            # The one-vs-rest path encodes labels directly as integers,
            # like fit(); use multiclass=True for string / non-integer
            # labels.
            try:
                classes = classes.astype(np.int32)
                y_enc = y.astype(np.int32)
            except (ValueError, TypeError):
                raise ValueError(
                    "non-integer labels are only supported with "
                    "multiclass=True (which fits a LabelEncoder); got "
                    "labels of dtype %s" % np.asarray(y).dtype)
            self._check_unknown_labels(y_enc, classes)
            binarizer_classes = classes

        # Binary OvR keeps the historical -1/+1 targets. For three or more
        # classes use 0/1 one-hot targets: with -1/+1 encoding a class that
        # is absent from the batch would be treated as the positive label
        # of every sample and get trained despite being absent.
        if len(binarizer_classes) <= 2:
            neg_label, pos_label = -1, 1
        else:
            neg_label, pos_label = 0, 1
        self.label_binarizer_ = LabelBinarizer(neg_label=neg_label,
                                               pos_label=pos_label)
        self.label_binarizer_.fit(binarizer_classes)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_classes = len(self.classes_)
        n_vectors = 1 if n_classes <= 2 else n_classes

        return y_enc, n_vectors

    def _encode_partial_fit_labels(self, y):
        """Encode y using the transformers fitted on the stable class set."""
        y = np.asarray(y)
        self._check_unknown_labels(y)

        if self.multiclass:
            y_enc = self.label_encoder_.transform(y).astype(np.int32)
        else:
            y_enc = y.astype(np.int32)

        return y_enc, self.coef_.shape[0]

    def _train(self, ds, y, n_vectors, rs, t_start):
        """Run the SGD solver over one batch and return the new step count."""
        n_samples = ds.get_n_samples()
        # Number of stochastic updates performed on this batch.
        max_iter = int(self.max_iter * n_samples)

        loss = self._get_loss()
        penalty = self._get_penalty()

        if n_vectors == 1 or not self.multiclass:
            Y = np.asfortranarray(self.label_binarizer_.transform(y),
                                  dtype=np.float64)
            t_end = t_start
            for i in range(n_vectors):
                t_end = _binary_sgd(self,
                                    self.coef_, self.intercept_, i,
                                    ds, Y[:, i], loss, penalty,
                                    self.alpha,
                                    self._get_learning_rate(),
                                    self.eta0, self.power_t,
                                    self.fit_intercept,
                                    self.intercept_decay,
                                    max_iter, self.shuffle, rs,
                                    self.callback, self.n_calls, self.verbose,
                                    t_start)
            return t_end

        else:
            return _multiclass_sgd(self, self.coef_, self.intercept_,
                                   ds, y.astype(np.int32), loss, penalty,
                                   self.alpha, self._get_learning_rate(),
                                   self.eta0, self.power_t, self.fit_intercept,
                                   self.intercept_decay,
                                   max_iter,
                                   self.shuffle, rs, self.callback,
                                   self.n_calls, self.verbose,
                                   t_start)

    def fit(self, X, y):
        """Fit model according to X and y.

        A call to ``fit`` always starts training from scratch: weights,
        intercept, learning-rate step counter and RNG state are reset.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : classifier
            Returns self.
        """
        rs = check_random_state(self.random_state)

        # Discard any state left over from a previous partial_fit.
        self._sgd_initialized_ = False
        self._sgd_t_ = 0
        self._sgd_random_state_ = None

        reencode = self.multiclass
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)

        ds = get_dataset(X)
        n_features = ds.get_n_features()
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        self._train(ds, y, n_vectors, rs, 0)

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        return self

    def partial_fit(self, X, y, classes=None):
        """Fit model on a single batch of samples.

        ``partial_fit`` performs one incremental training pass over the
        batch and keeps the model state between calls: weights, intercept,
        the learning-rate step counter (so that 'pegasos' and 'invscaling'
        schedules continue instead of restarting) and the RNG state are
        preserved. Delayed regularization (the L1 / l1/l2 penalty buffers
        and the L2 weight scale) is flushed into the weights at the end of
        each call, so no penalty state is lost between batches.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            Training vectors. Sparse input is kept sparse.

        y : array-like, shape = [n_samples]
            Target values. Every label must belong to ``classes`` as
            declared on the first call; unknown labels raise a
            ``ValueError`` and are never silently numbered.

        classes : array-like, shape = [n_classes]
            All classes that can ever be encountered by the stream. Must
            be provided on the first call (it fixes the label encoding and
            model size); ignored on subsequent calls. Classes absent from
            a batch keep their slot and are not renumbered.

        Returns
        -------
        self : classifier
            Returns self.
        """
        ds = get_dataset(X)
        n_features = ds.get_n_features()

        if not getattr(self, "_sgd_initialized_", False):
            if classes is None:
                raise ValueError(
                    "classes must be passed on the first call to "
                    "partial_fit to declare the stable set of labels.")

            y, n_vectors = self._init_partial_fit_labels(y, classes)

            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
            self._sgd_t_ = 0
            # One RNG, reused by every subsequent call, so shuffle draws
            # continue the same random stream.
            self._sgd_random_state_ = check_random_state(self.random_state)
        else:
            if n_features != self.coef_.shape[1]:
                raise ValueError(
                    "X has %d features, but this SGDClassifier was "
                    "initialized with %d features."
                    % (n_features, self.coef_.shape[1]))

            y, n_vectors = self._encode_partial_fit_labels(y)

        self._sgd_t_ = self._train(ds, y, n_vectors,
                                   self._sgd_random_state_, self._sgd_t_)
        self._sgd_initialized_ = True

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        return self


class SGDRegressor(BaseRegressor, _BaseSGD):
    r"""Estimator for learning linear regressors by SGD.

    Parameters
    ----------
    loss : str, 'squared', 'epsilon_insensitive', 'huber'
        Loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    alpha : float
        Weight of the penalty term.

    learning_rate : 'pegasos', 'constant', 'invscaling'
        Learning schedule to use.

    eta0 : float
        Step size.

    power_t : float
        Power to be used (when learning_rate='invscaling').

    epsilon : float
        Value to be used for epsilon-insensitive loss.

    fit_intercept : bool
        Whether to fit the intercept or not.

    intercept_decay : float
        Value by which the intercept is multiplied (to regularize it).

    max_iter : int
        Maximum number of iterations to perform.

    shuffle : bool
        Whether to shuffle data.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.
    """

    def __init__(self, loss="squared", penalty="l2",
                 alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100, verbose=0):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.coef_ = None

    def _get_loss(self):
        losses = {
            "squared": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        Returns
        -------
        self : regressor
            Returns self.
        """
        rs = check_random_state(self.random_state)

        ds = get_dataset(X)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y)
        n_vectors = Y.shape[1]
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()

        for k in range(n_vectors):
            _binary_sgd(self,
                        self.coef_, self.intercept_, k,
                        ds, Y[:, k], loss, penalty, self.alpha,
                        self._get_learning_rate(),
                        self.eta0, self.power_t,
                        self.fit_intercept,
                        self.intercept_decay,
                        int(self.max_iter * n_samples), self.shuffle, rs,
                        self.callback, self.n_calls, self.verbose)

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        return self

    def predict(self, X):
        """
        Perform regression on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p : array, shape = [n_samples]
            Predicted target values for X
        """
        try:
            assert_all_finite(self.coef_)
            pred = safe_sparse_dot(X, self.coef_.T)
            pred += self.intercept_
        except ValueError:
            n_samples = X.shape[0]
            n_vectors = self.coef_.shape[0]
            pred = np.zeros((n_samples, n_vectors))

        if not self.outputs_2d_:
            pred = pred.ravel()

        return pred
