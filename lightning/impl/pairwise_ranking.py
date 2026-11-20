"""
==========================================
Pairwise learning-to-rank on query groups
==========================================

Relevance labels in a search log are only meaningful *relative to the query
they belong to*.  A document graded ``2`` for one query may be worse than a
document graded ``1`` for another query.  This module therefore turns the
problem into a collection of within-query pairwise decisions:

* for every query, documents with different relevance grades are paired;
* each pair yields the directed feature difference ``x_high - x_low`` and,
  for training a binary linear classifier with a symmetric loss, its reverse
  ``x_low - x_high`` with the opposite label;
* documents with the same grade are never paired and a pair never crosses a
  query boundary;
* the number of pairs created for a query can be bounded so that a query with
  many documents does not dominate the training set.

The learned linear coefficients are then applied to the *original* documents,
and :meth:`PairwiseRanker.rank` sorts the documents stably inside each query.

Both dense and sparse feature matrices are supported.  For sparse input the
constructed differences stay sparse.
"""

# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array

from .base import BaseEstimator
from .sgd import SGDClassifier


def _as_index_array(query_id, n_samples):
    query_id = np.asarray(query_id)
    if query_id.ndim != 1:
        raise ValueError("query_id must be a one-dimensional array.")
    if query_id.shape[0] != n_samples:
        raise ValueError("query_id has %d elements but X has %d samples."
                         % (query_id.shape[0], n_samples))
    return query_id


def _encode_relevance(y, relevance_order=None):
    """Return the encoded grades and the ordered grade values."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")

    if relevance_order is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(relevance_order)
        if classes.ndim != 1:
            raise ValueError("relevance_order must be one-dimensional.")
        if np.unique(classes).shape[0] != classes.shape[0]:
            raise ValueError("relevance_order must not contain duplicate "
                             "grades.")
        missing = [label for label in np.unique(y)
                   if label not in set(classes.tolist())]
        if missing:
            raise ValueError("relevance_order does not contain all grades "
                             "present in y: %s" % missing)

    mapping = {label: rank for rank, label in enumerate(classes.tolist())}
    try:
        y_rank = np.asarray([mapping[label] for label in y.tolist()],
                            dtype=np.int32)
    except KeyError as exc:
        raise ValueError("y contains a grade not present in "
                         "relevance_order: %s" % exc) from exc

    return y_rank, classes


def _sample_without_replacement(n_total, n_selected, rng):
    """Select ``n_selected`` distinct integers from ``range(n_total)``."""
    if n_selected == n_total:
        return np.arange(n_total, dtype=np.int64)

    # sklearn chooses between set-based, reservoir and permutation sampling,
    # so this stays cheap both when the budget is tiny relative to the number
    # of possible pairs and when it is close to it.
    selected = sample_without_replacement(n_total, n_selected,
                                          method="auto",
                                          random_state=rng)
    return np.asarray(selected, dtype=np.int64)


def _allocate_budget(counts, budget):
    """Split ``budget`` across blocks proportionally, at least one each.

    A pure proportional split can starve a transition between two rare grades
    when another transition is very frequent.  Giving every non-empty block
    at least one pair keeps all grade transitions represented.
    """
    counts = np.asarray(counts, dtype=np.int64)
    n_blocks = counts.shape[0]
    allocation = np.zeros(n_blocks, dtype=np.int64)
    if budget < n_blocks:
        # Not enough budget to cover every block: give it to the largest
        # blocks first, deterministically by block order on ties.
        order = np.argsort(-counts, kind="stable")
        allocation[order[:budget]] = 1
        return allocation

    allocation[:] = 1
    remaining = budget - n_blocks
    if remaining == 0:
        return allocation

    available = counts - 1
    if available.sum() == 0:
        # Every block already has exactly one pair: nothing else can be
        # allocated without exceeding the available pairs.
        return allocation
    quota = remaining * (available / available.sum())
    extra = np.floor(quota).astype(np.int64)
    # Never exceed the number of pairs available in a block.
    extra = np.minimum(extra, available)
    allocation += extra
    left = budget - int(allocation.sum())

    # Distribute the rounding remainder by largest fractional part, cycling
    # until the budget is exhausted (some blocks may already be at their
    # capacity and cannot receive more pairs).
    fractions = quota - np.floor(quota)
    order = np.argsort(-fractions, kind="stable")
    while left > 0:
        progress = False
        for block in order:
            if left == 0:
                break
            if allocation[block] < counts[block]:
                allocation[block] += 1
                left -= 1
                progress = True
        if not progress:
            # Should only be reachable if budget > total, which the caller
            # guards against.  Stay safe and stop instead of looping.
            break

    return allocation


def _pairs_for_query(groups, max_pairs, rng):
    """Construct pairs for one query.

    ``groups`` maps an encoded grade to the document indices with that grade.
    A returned pair is ``(high_index, low_index)``, i.e. the first document has
    the higher relevance grade.  Same-grade documents are never paired.
    """
    grades = sorted(groups)
    blocks = []
    total = 0

    for high_grade in grades:
        for low_grade in grades:
            if high_grade <= low_grade:
                continue
            n_high = len(groups[high_grade])
            n_low = len(groups[low_grade])
            if n_high and n_low:
                count = n_high * n_low
                blocks.append((high_grade, low_grade, n_high, n_low, count))
                total += count

    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    if max_pairs is None or max_pairs >= total:
        allocations = [block[4] for block in blocks]
    else:
        allocations = _allocate_budget([block[4] for block in blocks],
                                       max_pairs)

    high_docs = []
    low_docs = []

    for block, allocation in zip(blocks, allocations):
        if allocation == 0:
            continue
        high_grade, low_grade, n_high, n_low, count = block
        if allocation >= count:
            offsets = np.arange(count, dtype=np.int64)
        else:
            offsets = _sample_without_replacement(count, allocation, rng)
            offsets.sort()
        high_docs.append(groups[high_grade][offsets // n_low])
        low_docs.append(groups[low_grade][offsets % n_low])

    if not high_docs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    return np.concatenate(high_docs), np.concatenate(low_docs)


def query_train_test_split(X, y, query_id, test_size=0.25,
                           random_state=None):
    """Split a ranking data set while keeping every query in one side.

    Unlike a plain sample-wise split, this function never puts two documents
    of the same query in different splits.  This is important for validation:
    relevance within a query is relative, so a document should not be used to
    train and evaluate the same query.

    Parameters
    ----------
    X : array-like or sparse matrix, shape = [n_samples, n_features]
        Feature matrix.

    y : array-like, shape = [n_samples]
        Relevance grades.

    query_id : array-like, shape = [n_samples]
        Query identifier of each document.

    test_size : float or int, default=0.25
        Fraction or absolute number of queries to put in the test set.

    random_state : int, RandomState or None
        Seed or random number generator.

    Returns
    -------
    X_train, X_test, y_train, y_test, query_train, query_test
        The split arrays.  The relative order of the samples inside each split
        is the same as in the input.
    """
    X = X if sp.issparse(X) else np.asarray(X)
    y = np.asarray(y)
    query_id = _as_index_array(query_id, X.shape[0])
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y have inconsistent numbers of samples.")

    unique_queries, inverse = np.unique(query_id, return_inverse=True)
    if unique_queries.shape[0] < 2:
        raise ValueError("At least two distinct queries are required to "
                         "create a train/test split.")

    from sklearn.model_selection import train_test_split

    train_query_indices, test_query_indices = train_test_split(
        np.arange(unique_queries.shape[0]),
        test_size=test_size,
        random_state=random_state)

    train_mask = np.isin(inverse, train_query_indices)
    test_mask = ~train_mask

    return (X[train_mask], X[test_mask],
            y[train_mask], y[test_mask],
            query_id[train_mask], query_id[test_mask])


class PairwiseRanker(BaseEstimator):
    r"""Pairwise linear ranker for query-grouped relevance data.

    The estimator builds directed, within-query feature differences and
    learns a linear direction with an existing :class:`SGDClassifier` loss.
    The learned ``coef_`` is then used to score original documents; ranking is
    performed inside each query by :meth:`rank`.

    Parameters
    ----------
    loss : str, default="hinge"
        Binary linear classification loss used on the difference vectors.
        Any loss supported by :class:`lightning.classification.SGDClassifier`
        for the binary setting can be used (e.g. ``"hinge"``,
        ``"squared_hinge"`` or ``"log"``).

    penalty : str, default="l2"
        Penalty used by the underlying linear classifier.

    alpha : float, default=1e-4
        Weight of the penalty term.

    learning_rate : str, default="invscaling"
        Learning schedule of the underlying SGD solver.  The default decays
        the step size over time, which is much more robust than a constant
        step size when feature differences have heterogeneous scales.

    eta0 : float, default=0.03
        Initial step size of the underlying SGD solver.

    power_t : float, default=0.5
        Power used by the ``"invscaling"`` learning schedule.

    max_iter : int, default=100
        Number of passes over the constructed difference vectors.

    max_pairs_per_query : int or None, default=1000
        Maximum number of unordered document pairs built for each query.  If a
        query has more valid pairs, a deterministic random subset is used.
        ``None`` means that all pairs are used (this can be very large).

    relevance_order : array-like or None, default=None
        Explicit ordering of relevance grades, from least to most relevant.
        By default the sorted unique values in ``y`` are used.

    shuffle : bool, default=True
        Whether the underlying SGD solver shuffles the difference vectors.

    random_state : int, RandomState or None, default=None
        Seed or random number generator used for pair sampling and by the
        underlying solver.

    verbose : int, default=0
        Verbosity level of the underlying solver.

    Attributes
    ----------
    coef_ : array, shape = [n_features]
        Learned ranking direction.  Larger values mean more relevant.

    classes_ : array, shape = [n_grades]
        Ordered relevance grades, from least to most relevant.

    query_ids_ : array, shape = [n_queries]
        Sorted unique query identifiers.

    pair_documents_ : array, shape = [n_pairs, 2]
        Original document indices for every pair.  Column 0 is the document
        with the higher grade and column 1 the document with the lower grade.

    pair_left_, pair_right_ : array, shape = [n_pairs]
        Convenience views of ``pair_documents_``.

    pair_differences_ : array or sparse matrix, shape = [n_pairs, n_features]
        Directed feature difference ``X[pair_left_] - X[pair_right_]``.  It is
        sparse whenever ``X`` is sparse.

    pair_labels_ : array, shape = [n_pairs]
        Labels for the directed differences (all equal to +1; the reverse
        examples are stored in ``training_differences_``).

    pair_queries_ : array, shape = [n_pairs]
        Query identifier of each pair.  Both documents of a pair always share
        this query.

    pair_query_indices_ : array, shape = [n_pairs]
        Encoded query index (into ``query_ids_``) of each pair.

    pair_relevance_ : array, shape = [n_pairs, 2]
        Relevance grade of the higher and lower document of each pair.

    training_differences_ : array or sparse matrix
        Shape ``[2 * n_pairs, n_features]``.  The actual design matrix passed
        to the linear classifier: each directed difference and its reverse.

    training_labels_ : array, shape = [2 * n_pairs]
        +1 for the directed difference and -1 for its reverse.

    training_pair_indices_ : array, shape = [2 * n_pairs]
        Index into ``pair_documents_`` for every training row.

    training_documents_ : array, shape = [2 * n_pairs, 2]
        Original document indices attached to every training row.

    training_directions_ : array, shape = [2 * n_pairs]
        +1 for the canonical (high, low) direction, -1 for the reverse row.

    n_pairs_ : int
        Number of unordered pairs created.

    n_pairs_per_query_ : array, shape = [n_queries]
        Number of pairs created for each query (in ``query_ids_`` order).

    n_features_in_ : int
        Number of features seen during fit.

    estimator_ : SGDClassifier
        The fitted linear classifier on the difference vectors.

    Examples
    --------
    >>> from lightning.ranking import PairwiseRanker
    >>> from lightning.ranking import query_train_test_split
    >>> Xtr, Xte, ytr, yte, qtr, qte = query_train_test_split(
    ...     X, y, query_id, random_state=0)
    >>> ranker = PairwiseRanker(max_pairs_per_query=500, random_state=0)
    >>> ranker.fit(Xtr, ytr, qtr)
    >>> rankings = ranker.rank(Xte, qte)
    >>> ranker.score(Xte, yte, qte)  # within-query pairwise accuracy
    """

    def __init__(self, loss="hinge", penalty="l2", alpha=1e-4,
                 learning_rate="invscaling", eta0=0.03, power_t=0.5,
                 max_iter=100, max_pairs_per_query=1000,
                 relevance_order=None, shuffle=True, random_state=None,
                 verbose=0):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.max_pairs_per_query = max_pairs_per_query
        self.relevance_order = relevance_order
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    def _build_pairs(self, y_rank, query_inverse, n_queries, rng):
        high_all = []
        low_all = []
        query_all = []
        counts = np.zeros(n_queries, dtype=np.int64)

        # Group the samples by query once.  Sorting the inverse mapping keeps
        # the original relative order of documents inside each query, which
        # makes the constructed pairs deterministic for a given seed.
        order = np.argsort(query_inverse, kind="stable")
        sorted_queries = query_inverse[order]
        boundaries = np.flatnonzero(np.diff(sorted_queries)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [sorted_queries.shape[0]]])

        for start, end in zip(starts, ends):
            in_query = order[start:end]
            if in_query.size < 2:
                continue
            query_index = int(query_inverse[in_query[0]])

            grades = y_rank[in_query]
            groups = {}
            for grade in np.unique(grades):
                groups[int(grade)] = in_query[grades == grade]

            high, low = _pairs_for_query(groups, self.max_pairs_per_query,
                                         rng)
            if high.size == 0:
                continue

            counts[query_index] = high.size
            high_all.append(high)
            low_all.append(low)
            query_all.append(np.full(high.size, query_index, dtype=np.int64))

        if not high_all:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int64), counts)

        return (np.concatenate(high_all), np.concatenate(low_all),
                np.concatenate(query_all), counts)

    def fit(self, X, y, query_id):
        """Fit the pairwise ranker.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            Document feature matrix.

        y : array-like, shape = [n_samples]
            Relevance grade of each document.  Grades are only compared
            between documents of the same query.

        query_id : array-like, shape = [n_samples]
            Query identifier of each document.

        Returns
        -------
        self : PairwiseRanker
            Returns self.
        """
        if self.max_pairs_per_query is not None:
            if not isinstance(self.max_pairs_per_query, (int, np.integer)):
                raise ValueError("max_pairs_per_query must be an integer "
                                 "or None.")
            if self.max_pairs_per_query < 1:
                raise ValueError("max_pairs_per_query must be positive.")

        X = check_array(X, accept_sparse="csr", dtype=np.float64,
                        ensure_min_samples=2)
        if sp.issparse(X):
            X = X.tocsr()
            X.sort_indices()
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")

        query_id = _as_index_array(query_id, X.shape[0])
        y_rank, classes = _encode_relevance(y, self.relevance_order)
        self.classes_ = classes

        query_values, query_inverse = np.unique(query_id, return_inverse=True)
        self.query_ids_ = query_values
        n_queries = query_values.shape[0]

        rng = check_random_state(self.random_state)
        high, low, query_index, counts = self._build_pairs(
            y_rank, query_inverse, n_queries, rng)

        if high.size == 0:
            raise ValueError("No valid within-query pairs were found. "
                             "At least one query needs two different "
                             "relevance grades.")

        # Preserve sample identity: every pair knows its two original rows.
        self.pair_left_ = high
        self.pair_right_ = low
        self.pair_documents_ = np.column_stack([high, low])
        self.pair_query_indices_ = query_index
        self.pair_queries_ = query_values[query_index]
        self.pair_relevance_ = np.column_stack([y[high], y[low]])
        self.pair_labels_ = np.ones(high.size, dtype=np.float64)
        self.n_pairs_ = int(high.size)
        self.n_pairs_per_query_ = counts

        # Directed sparse (or dense) feature differences.  A pair with
        # high relevance has difference X[high] - X[low]; the classifier
        # should assign it a positive score.
        differences = X[high] - X[low]
        if sp.issparse(differences):
            differences = differences.tocsr()
        else:
            differences = np.asarray(differences, dtype=np.float64)
        self.pair_differences_ = differences

        # A symmetric binary loss needs examples of both classes.  Add the
        # reverse direction as a negative example.  The original pair index is
        # kept for both rows.
        reverse = -differences
        if sp.issparse(differences):
            training_differences = sp.vstack(
                [differences, reverse], format="csr")
        else:
            training_differences = np.vstack([differences, reverse])
        training_labels = np.concatenate([
            np.ones(self.n_pairs_, dtype=np.float64),
            -np.ones(self.n_pairs_, dtype=np.float64)
        ])
        training_pair_indices = np.concatenate([
            np.arange(self.n_pairs_, dtype=np.int64),
            np.arange(self.n_pairs_, dtype=np.int64)
        ])

        self.training_differences_ = training_differences
        self.training_labels_ = training_labels
        self.training_pair_indices_ = training_pair_indices
        # Sample identity for every row of the design matrix: the two
        # original documents, and the direction (+1 means the row is
        # ``x_high - x_low``, -1 its reverse).
        self.training_documents_ = self.pair_documents_[training_pair_indices]
        directions = np.where(training_labels > 0, 1, -1).astype(np.int8)
        self.training_directions_ = directions
        self.n_features_in_ = X.shape[1]

        estimator = SGDClassifier(
            loss=self.loss,
            penalty=self.penalty,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            power_t=self.power_t,
            max_iter=self.max_iter,
            fit_intercept=False,  # differences are centered by construction
            shuffle=self.shuffle,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        estimator.fit(training_differences, training_labels)

        coef = np.asarray(estimator.coef_).ravel()
        if not np.all(np.isfinite(coef)):
            raise ValueError(
                "The pairwise ranker produced non-finite coefficients. "
                "This usually means that the SGD step size is too large for "
                "the scale of the difference vectors; try a smaller eta0 or "
                "learning_rate='invscaling' with a smaller eta0.")

        self.estimator_ = estimator
        self.coef_ = coef.copy()

        return self

    def decision_function(self, X):
        """Return the relevance score of each original document.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]

        Returns
        -------
        scores : array, shape = [n_samples]
            Larger values indicate more relevant documents.  Scores are only
            comparable between documents of the same query.
        """
        if not hasattr(self, "coef_"):
            raise ValueError("This PairwiseRanker instance is not fitted yet.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features, but PairwiseRanker was "
                             "fitted with %d features."
                             % (X.shape[1], self.n_features_in_))
        scores = safe_sparse_dot(X, self.coef_)
        return np.asarray(scores, dtype=np.float64).ravel()

    def predict(self, X):
        """Alias for :meth:`decision_function` (returns relevance scores)."""
        return self.decision_function(X)

    def _grouped_indices(self, query_id, n_samples):
        query_id = _as_index_array(query_id, n_samples)
        groups = {}
        for index, query in enumerate(query_id.tolist()):
            groups.setdefault(query, []).append(index)
        return groups

    def _rank_from_scores(self, scores, query_id, descending=True,
                          top_k=None):
        if scores.shape[0] != len(query_id):
            raise ValueError("X and query_id have inconsistent numbers of "
                             "samples.")
        groups = self._grouped_indices(query_id, scores.shape[0])

        rankings = {}
        for query, indices in groups.items():
            indices = np.asarray(indices, dtype=np.int64)
            # A stable sort keeps the original order for equal scores.
            order = np.argsort(-scores[indices] if descending
                               else scores[indices], kind="stable")
            ranked = indices[order]
            if top_k is not None:
                ranked = ranked[:top_k]
            rankings[query] = ranked
        return rankings

    def rank(self, X, query_id, descending=True, top_k=None):
        """Score original documents and rank them stably within each query.

        Ties are broken by the original document order, so the result is
        deterministic and independent of the sorting implementation.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            Original document features (not difference vectors).

        query_id : array-like, shape = [n_samples]
            Query identifier of each document.

        descending : bool, default=True
            If True, most relevant documents come first.

        top_k : int or None, default=None
            If given, keep only the first ``top_k`` documents of each query.

        Returns
        -------
        rankings : dict
            Maps each query identifier to an array of original document
            indices, ordered within that query.
        """
        scores = self.decision_function(X)
        return self._rank_from_scores(scores, query_id,
                                      descending=descending, top_k=top_k)

    def rank_scores(self, X, query_id, descending=True, top_k=None):
        """Like :meth:`rank`, but also return the score of each document.

        Returns
        -------
        rankings : dict
            Maps each query identifier to ``(indices, scores)``.
        """
        scores = self.decision_function(X)
        rankings = self._rank_from_scores(scores, query_id,
                                          descending=descending,
                                          top_k=top_k)
        return {query: (indices, scores[indices])
                for query, indices in rankings.items()}

    def score(self, X, y, query_id):
        """Return within-query pairwise ranking accuracy.

        For every pair of documents of the same query with different grades,
        the pair is correct when the higher-graded document receives a higher
        score.  Exact score ties count as half a correct pair.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]

        y : array-like, shape = [n_samples]
            True relevance grades.

        query_id : array-like, shape = [n_samples]

        Returns
        -------
        accuracy : float
            Pairwise accuracy in [0, 1].
        """
        y = np.asarray(y)
        scores = self.decision_function(X)
        if scores.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")
        query_id = _as_index_array(query_id, scores.shape[0])

        classes = self.classes_
        mapping = {label: rank for rank, label in enumerate(classes.tolist())}
        try:
            y_rank = np.asarray([mapping[label] for label in y.tolist()],
                                dtype=np.int32)
        except KeyError as exc:
            raise ValueError("y contains a grade unseen during fit: %s"
                             % exc) from exc

        correct = 0.0
        total = 0

        for query in np.unique(query_id):
            in_query = np.flatnonzero(query_id == query)
            if in_query.size < 2:
                continue
            query_grades = y_rank[in_query]
            query_scores = scores[in_query]

            for high_grade in range(len(classes)):
                high = query_scores[query_grades == high_grade]
                if high.size == 0:
                    continue
                high.sort()

                for low_grade in range(high_grade):
                    low = query_scores[query_grades == low_grade]
                    if low.size == 0:
                        continue
                    # Number of high-scored documents strictly greater than
                    # each low-scored document, plus half of the exact ties.
                    right = np.searchsorted(high, low, side="right")
                    left = np.searchsorted(high, low, side="left")
                    greater = high.size - right
                    equal = right - left
                    correct += float(greater.sum()) + 0.5 * float(equal.sum())
                    total += high.size * low.size

        if total == 0:
            raise ValueError("No comparable within-query pairs are present "
                             "in y.")
        return correct / total


# Alternative names kept as aliases for discoverability / compatibility.
PairwiseRankingClassifier = PairwiseRanker
PairwiseRanking = PairwiseRanker
