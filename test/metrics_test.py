import pytest

from ann_benchmarks.plotting.metrics import knn, queries_per_second, index_size, build_time, candidates


class DummyMetric:
    def __init__(self):
        self.attrs = {}
        self.d = {}

    def __getitem__(self, key):
        return self.d.get(key, None)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __contains__(self, key):
        return key in self.d

    def create_group(self, name):
        self.d[name] = DummyMetric()
        return self.d[name]


def test_recall():
    exact_queries = [[1, 2]]
    run1 = [[]]
    run2 = [[1, 3]]
    run3 = [[2]]
    run4 = [[2, 1]]

    assert knn(exact_queries, run1, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.0)
    assert knn(exact_queries, run2, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.5)
    assert knn(exact_queries, run3, 2, DummyMetric()).attrs["mean"] == pytest.approx(0.5)
    assert knn(exact_queries, run4, 2, DummyMetric()).attrs["mean"] == pytest.approx(1.0)


def test_queries_per_second():
    assert queries_per_second([0.01, 0.015, 0.005]) == 100


def test_index_size():
    assert index_size({"index_size": 100}) == 100


def test_build_time():
    assert build_time({"build_time": 100}) == 100


def test_candidates():
    assert candidates({"candidates": 10}) == 10
