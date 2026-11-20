import numpy as np
import pytest
import scipy.sparse as sp

from lightning.ranking import PairwiseRanker
from lightning.ranking import PairwiseRankingClassifier
from lightning.ranking import query_train_test_split


def _make_query_data(n_queries=12, n_docs=20, n_features=6, n_grades=3,
                     noise=0.35, random_state=0):
    """Synthetic data whose relevance is only meaningful within a query.

    Documents of a query are graded by their relative position on a latent
    feature.  That feature is shifted by a query-specific constant, so its raw
    value has no global relationship with the grade: comparing documents from
    different queries is meaningless.  A valid model must therefore learn from
    within-query differences only.
    """
    rng = np.random.RandomState(random_state)
    X = []
    y = []
    queries = []
    for query in range(n_queries):
        latent = rng.normal(size=n_docs)
        # Evenly spread grades over the within-query ranking of `latent`.
        order = np.argsort(np.argsort(latent))
        grades = (order * n_grades) // n_docs

        features = rng.normal(size=(n_docs, n_features))
        # The informative feature carries a strong query-specific offset.
        # Thresholding it globally is therefore useless.
        features[:, 0] = 20 * query + latent + noise * rng.normal(size=n_docs)
        # An extra feature that is pure noise.
        features[:, 1] = rng.normal(size=n_docs)
        X.append(features)
        y.extend(grades.tolist())
        queries.extend([query] * n_docs)
    return np.vstack(X), np.asarray(y), np.asarray(queries)


@pytest.fixture(scope="module")
def ranking_data():
    return _make_query_data()


def test_pairwise_ranker_fit_scores_and_ranks(ranking_data):
    X, y, query_id = ranking_data
    est = PairwiseRanker(max_pairs_per_query=100, max_iter=50,
                         random_state=0)
    est.fit(X, y, query_id)

    assert est.coef_.shape == (X.shape[1],)
    assert np.all(np.isfinite(est.coef_))
    assert est.n_pairs_ > 0
    np.testing.assert_array_equal(est.pair_left_, est.pair_documents_[:, 0])
    np.testing.assert_array_equal(est.pair_right_, est.pair_documents_[:, 1])

    # Ranking must be materially better than chance on independent-looking
    # data, and the entry point must return actual per-query orders.
    assert est.score(X, y, query_id) > 0.85

    rankings = est.rank(X, query_id)
    assert set(rankings) == set(query_id.tolist())

    scores = est.decision_function(X)
    for query, indices in rankings.items():
        in_query = np.flatnonzero(query_id == query)
        np.testing.assert_array_equal(np.sort(indices), in_query)
        query_scores = scores[indices]
        assert np.all(np.diff(query_scores) <= 0)

    top = est.rank(X, query_id, top_k=3)
    assert all(len(indices) == 3 for indices in top.values())

    ranked_scores = est.rank_scores(X, query_id)
    for query, (indices, values) in ranked_scores.items():
        np.testing.assert_array_equal(values, scores[indices])


def test_pairwise_ranker_pairs_do_not_cross_or_match_grades(ranking_data):
    X, y, query_id = ranking_data
    est = PairwiseRanker(max_pairs_per_query=None, random_state=0)
    est.fit(X, y, query_id)

    # Same query for both members of every pair.
    assert np.all(query_id[est.pair_left_] == query_id[est.pair_right_])
    for left, right in est.pair_documents_:
        assert query_id[left] == query_id[right]
        # Directed: the left document must have the higher grade.
        assert y[left] > y[right]

    # And explicitly: no same-grade pair and no cross-query pair exists.
    for left, right in est.pair_documents_:
        assert y[left] != y[right]

    # Every pair is stored once in pair_documents_, plus its reverse row in
    # the design matrix, with identical sample identity information.
    assert est.training_differences_.shape[0] == 2 * est.n_pairs_
    np.testing.assert_array_equal(
        est.training_pair_indices_[:est.n_pairs_],
        est.training_pair_indices_[est.n_pairs_:])
    np.testing.assert_array_equal(est.pair_differences_,
                                  X[est.pair_left_] - X[est.pair_right_])
    np.testing.assert_array_equal(est.training_documents_[:est.n_pairs_],
                                  est.pair_documents_)
    np.testing.assert_array_equal(est.training_directions_[:est.n_pairs_],
                                  np.ones(est.n_pairs_))
    np.testing.assert_array_equal(est.training_differences_[:est.n_pairs_],
                                  est.pair_differences_)
    np.testing.assert_array_equal(
        est.training_differences_[est.n_pairs_:],
        -est.pair_differences_)


def test_pairwise_ranker_respects_pair_budget():
    # One query with 10 documents at grade 0, 10 at grade 1 and 10 at grade 2
    # can produce 300 unordered cross-grade pairs.
    rng = np.random.RandomState(0)
    X = rng.normal(size=(30, 4))
    X[:, 0] = np.repeat([0, 1, 2], 10) + 0.1 * X[:, 0]
    y = np.repeat([0, 1, 2], 10)
    query_id = np.zeros(30, dtype=int)

    est = PairwiseRanker(max_pairs_per_query=37, random_state=0)
    est.fit(X, y, query_id)
    assert est.n_pairs_ == 37
    assert est.n_pairs_per_query_[0] == 37
    # Every grade transition is still represented under a tight budget.
    transitions = set(zip(y[est.pair_left_].tolist(),
                          y[est.pair_right_].tolist()))
    assert transitions == {(1, 0), (2, 0), (2, 1)}

    est = PairwiseRanker(max_pairs_per_query=None, random_state=0)
    est.fit(X, y, query_id)
    assert est.n_pairs_ == 300

    # A budget smaller than the number of transitions keeps only as many
    # transitions as it can afford, instead of silently exceeding the budget.
    est = PairwiseRanker(max_pairs_per_query=2, random_state=0)
    est.fit(X, y, query_id)
    assert est.n_pairs_ == 2

    # Pair construction is deterministic for a given seed.
    est_a = PairwiseRanker(max_pairs_per_query=37, random_state=3).fit(
        X, y, query_id)
    est_b = PairwiseRanker(max_pairs_per_query=37, random_state=3).fit(
        X, y, query_id)
    np.testing.assert_array_equal(est_a.pair_documents_,
                                  est_b.pair_documents_)


def test_pairwise_ranker_sparse_input(ranking_data):
    X, y, query_id = ranking_data
    X_sparse = sp.csr_matrix(X)
    X_sparse[X_sparse < 0] = 0
    X_sparse.eliminate_zeros()

    est = PairwiseRanker(max_pairs_per_query=50, max_iter=50,
                         random_state=0)
    est.fit(X_sparse, y, query_id)
    assert sp.issparse(est.pair_differences_)
    assert sp.issparse(est.training_differences_)
    rankings = est.rank(X_sparse, query_id)
    assert set(rankings) == set(query_id.tolist())
    assert est.score(X_sparse, y, query_id) > 0.75


def test_pairwise_ranker_relevance_order_and_errors(ranking_data):
    X, y, query_id = ranking_data

    # A custom grade order must be respected when constructing pairs.
    order = [2, 1, 0]
    est = PairwiseRanker(max_pairs_per_query=10, relevance_order=order,
                         random_state=0)
    est.fit(X, y, query_id)
    np.testing.assert_array_equal(est.classes_, order)
    for left, right in est.pair_documents_:
        # Grade 2 is the most relevant in this custom order.
        assert order.index(y[left]) > order.index(y[right])

    with pytest.raises(ValueError):
        PairwiseRanker(relevance_order=[0, 1]).fit(X, y, query_id)

    with pytest.raises(ValueError):
        PairwiseRanker(relevance_order=[0, 1, 2, 2]).fit(X, y, query_id)

    with pytest.raises(ValueError):
        est.score(X, y + 10, query_id)

    with pytest.raises(ValueError):
        pairwise = PairwiseRanker()
        pairwise.decision_function(X)

    with pytest.raises(ValueError):
        # Only one grade anywhere: nothing to learn.
        PairwiseRanker().fit(X, np.zeros_like(y), query_id)

    with pytest.raises(ValueError):
        PairwiseRanker().fit(X, y, query_id[:-1])


def test_query_train_test_split_isolates_queries(ranking_data):
    X, y, query_id = ranking_data
    X_train, X_test, y_train, y_test, q_train, q_test = (
        query_train_test_split(X, y, query_id, test_size=0.25,
                               random_state=0))

    assert X_train.shape[0] == y_train.shape[0] == q_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0] == q_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == X.shape[0]

    # Queries are isolated between the two splits.
    assert not (set(q_train.tolist()) & set(q_test.tolist()))
    assert set(q_train.tolist()) | set(q_test.tolist()) == \
        set(query_id.tolist())

    # Training only on the training queries and evaluating on the test
    # queries works end to end.
    est = PairwiseRanker(max_pairs_per_query=100, random_state=0)
    est.fit(X_train, y_train, q_train)
    rankings = est.rank(X_test, q_test)
    assert set(rankings) == set(q_test.tolist())
    assert est.score(X_test, y_test, q_test) > 0.8


def test_query_train_test_split_sparse_and_clone(ranking_data):
    X, y, query_id = ranking_data
    X_sparse = sp.csr_matrix(np.abs(X))

    # The split preserves sparse format and still isolates queries.
    splits = query_train_test_split(X_sparse, y, query_id, test_size=0.3,
                                    random_state=0)
    X_train, X_test, y_train, y_test, q_train, q_test = splits
    assert sp.issparse(X_train) and sp.issparse(X_test)
    assert not (set(q_train.tolist()) & set(q_test.tolist()))

    # Cloning with the same seed reproduces the model, which is what
    # grid-search / cross-validation relies on.
    from sklearn.base import clone

    est = PairwiseRanker(max_pairs_per_query=50, random_state=0)
    est.fit(X_train, y_train, q_train)
    cloned = clone(est).fit(X_train, y_train, q_train)
    np.testing.assert_allclose(est.coef_, cloned.coef_)


def test_pairwise_ranker_group_kfold_validation(ranking_data):
    # GroupKFold never splits a query across folds, mirroring how a ranking
    # model should be validated.
    from sklearn.model_selection import GroupKFold

    X, y, query_id = ranking_data
    fold_scores = []
    for train, test in GroupKFold(n_splits=3).split(X, y, query_id):
        assert not (set(query_id[train].tolist()) &
                    set(query_id[test].tolist()))
        est = PairwiseRanker(max_pairs_per_query=80, random_state=0)
        est.fit(X[train], y[train], query_id[train])
        fold_scores.append(est.score(X[test], y[test], query_id[test]))

    assert np.mean(fold_scores) > 0.8


def test_pairwise_ranker_alias():
    assert PairwiseRankingClassifier is PairwiseRanker
