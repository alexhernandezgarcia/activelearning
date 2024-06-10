import pytest
from activelearning.dataset.grid import GridData
import torch


@pytest.fixture
def grid_data_1():
    X_data = torch.tensor([[0, 0], [0, 1], [1, 0]])
    y_data = torch.tensor([2.0, 1.5, 0.7])
    return GridData(grid_size=10, X_data=X_data, y_data=y_data)


@pytest.fixture
def grid_data_2():
    X_data = torch.tensor([[1, 0], [2, 1], [1, 7]])
    y_data = torch.tensor([2.1, 0.5, 0.9])
    return GridData(grid_size=3, X_data=X_data, y_data=y_data)


@pytest.mark.parametrize(
    "grid_data",
    [
        "grid_data_1",
        "grid_data_2",
    ],
)
def test__grid_data__initializes_properly(grid_data, request):
    grid_data = request.getfixturevalue(grid_data)
    assert True


@pytest.mark.parametrize(
    "states, states2proxy",
    [
        (
            torch.Tensor([[0, 0], [5, 5], [10, 10]]),
            torch.Tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
        ),
    ],
)
def test__grid_data__proxy(states, states2proxy):
    data = GridData(grid_size=10, X_data=states, y_data=None)
    x = data[:]
    assert torch.equal(states2proxy, x)


@pytest.mark.parametrize(
    "score, score_norm",
    [
        (
            torch.Tensor([0.8709, 171.1192, 273.2218, 174.8875]),
            torch.Tensor(
                [
                    0.0,
                    0.6251064538955688,
                    1.0,
                    0.6389426589012146,
                ]
            ),
        ),
    ],
)
def test__grid_data__target_normalization(score, score_norm):
    X_data = torch.tensor([[0, 0], [0, 1], [1, 0]])
    data = GridData(
        grid_size=10,
        X_data=X_data,
        y_data=score,
        normalize_scores=True,
        float=torch.float32,
    )
    _, y = data[:]
    assert torch.equal(y, score_norm)
