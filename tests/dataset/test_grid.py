import pytest
from activelearning.dataset.grid import (
    GridData,
    BraninDatasetHandler,
    HartmannDatasetHandler,
)
import torch
from gflownet.envs.grid import Grid as GridEnv
import numpy as np


@pytest.fixture
def grid_data_1():
    X_data = torch.tensor([[0, 0], [0, 1], [1, 0]])
    y_data = torch.tensor([2.0, 1.5, 0.7])
    return GridData(X_data=X_data, y_data=y_data)


@pytest.fixture
def grid_data_2():
    X_data = torch.tensor([[1, 0], [2, 1], [1, 7]])
    y_data = torch.tensor([2.1, 0.5, 0.9])
    return GridData(X_data=X_data, y_data=y_data)


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
            torch.Tensor(
                [
                    [0, 0],
                    [5, 5],
                    [9, 9],
                ]
            ),
            torch.Tensor(
                [
                    [-1.0, -1.0],
                    [0.1111111111111111, 0.1111111111111111],
                    [1.0, 1.0],
                ]
            ),
        ),
    ],
)
def test__grid_data__state2proxy(states, states2proxy):
    env = GridEnv(length=10)
    data = GridData(X_data=states, y_data=None, state2result=env.states2proxy)
    x = data[:]
    assert torch.equal(states2proxy, x)


@pytest.mark.parametrize(
    "score, score_norm",
    [
        (
            torch.Tensor(
                [
                    0.8709,
                    171.1192,
                    273.2218,
                    174.8875,
                ]
            ),
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
        X_data=X_data,
        y_data=score,
        normalize_scores=True,
        float=torch.float32,
    )
    _, y = data[:]
    assert torch.equal(y, score_norm)


@pytest.fixture
def branin_dataset_handler():
    env = GridEnv(length=10)
    dataset_handler = BraninDatasetHandler(
        env=env,
        train_path="./data/branin/data_%i_train.csv" % env.length,
        train_fraction=1.0,
        float_precision=32,
    )
    return dataset_handler


def test__branin__initializes_properly(branin_dataset_handler):
    branin_dataset_handler.get_dataloader()
    branin_dataset_handler.get_candidate_set()
    assert True


def test__branin__state2proxy(branin_dataset_handler):
    train_loader, _ = branin_dataset_handler.get_dataloader()
    x, y = train_loader.dataset[:]
    assert torch.equal(
        x,
        torch.Tensor(
            [
                [-1.0000, -1.0000],
                [-1.0000, -0.1111111111111111],
                [-0.1111111111111111, -1.0000],
                [0.7777777910232544, 0.7777777910232544],
            ]
        ),
    )


@pytest.mark.parametrize(
    "states, states2proxy",
    [
        (
            torch.Tensor(
                [
                    [0, 0],
                    [5, 5],
                    [9, 9],
                ]
            ),
            torch.Tensor(
                [
                    [-1.0, -1.0],
                    [0.1111111111111111, 0.1111111111111111],
                    [1.0, 1.0],
                ]
            ),
        ),
    ],
)
def test__branin__states2proxy(branin_dataset_handler, states, states2proxy):
    proxy_states = branin_dataset_handler.env.states2proxy(states)
    assert torch.equal(states2proxy, proxy_states)


@pytest.mark.parametrize(
    "new_states, all_states, new_scores, all_scores",
    [
        (
            torch.Tensor(
                [
                    [0, 0],
                    [5, 5],
                    [9, 9],
                ]
            ),
            torch.Tensor(
                [
                    [0, 0],
                    [0, 4],
                    [4, 0],
                    [8, 8],
                    [0, 0],
                    [5, 5],
                    [9, 9],
                ]
            ),
            torch.Tensor(
                [
                    100.0,
                    200.0,
                    300.0,
                ]
            ),
            torch.Tensor(
                [
                    0.8709,
                    171.1192,
                    273.2218,
                    174.8875,
                    100.0,
                    200.0,
                    300.0,
                ]
            ),
        ),
    ],
)
def test__branin__update_dataset(
    branin_dataset_handler, new_states, all_states, new_scores, all_scores
):
    branin_dataset_handler.update_dataset(new_states, new_scores)
    assert torch.equal(branin_dataset_handler.train_data.X_data, all_states)
    assert torch.equal(branin_dataset_handler.train_data.y_data, all_scores)


@pytest.mark.parametrize(
    "readable_state, state",
    [
        (
            [
                "[0 0]",
                "[0 4]",
                "[4 0]",
                "[8 8]",
            ],
            [
                [0, 0],
                [0, 4],
                [4, 0],
                [8, 8],
            ],
        )
    ],
)
def test__branin__state2readable(branin_dataset_handler, readable_state, state):
    test_state = [
        branin_dataset_handler.env.readable2state(sample) for sample in readable_state
    ]
    assert test_state == state

    test_readable_state = [
        branin_dataset_handler.env.state2readable(sample) for sample in test_state
    ]
    assert readable_state == test_readable_state


def test__branin__candidate_set(branin_dataset_handler):
    candidate_set, xi, yi = branin_dataset_handler.get_candidate_set()
    assert np.all(np.equal(xi, yi))
    assert np.all(
        np.equal(
            xi,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
    )


@pytest.fixture
def hartmann_dataset_handler():
    env = GridEnv(n_dim=6, length=10)
    dataset_handler = HartmannDatasetHandler(
        env=env,
        train_path="./data/hartmann/data_train.csv",
        train_fraction=1.0,
        float_precision=32,
    )
    return dataset_handler


@pytest.mark.parametrize(
    "indices, states",
    [
        (
            [0, 100, 438, 10**6 - 1],
            torch.Tensor(
                [
                    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -0.7777777910232544, -1.0, -1.0],
                    [
                        -1.0,
                        -1.0,
                        -1.0,
                        -0.1111111119389534,
                        -0.3333333432674408,
                        0.7777777910232544,
                    ],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
        )
    ],
)
def test__hartmann__candidate_set(hartmann_dataset_handler, indices, states):
    candidate_set, xi, yi = hartmann_dataset_handler.get_candidate_set()
    assert np.all(np.equal(xi, yi))
    assert np.all(
        np.equal(
            xi,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        )
    )
    test_states = candidate_set.dataset[indices]
    assert torch.equal(states, test_states)
