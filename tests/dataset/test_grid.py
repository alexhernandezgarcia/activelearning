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
