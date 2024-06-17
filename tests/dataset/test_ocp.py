import pytest
import torch
from activelearning.dataset.ocp import OCPDatasetHandler
from gflownet.envs.crystals.surface import CrystalSurface as CrystalSurfaceEnv


@pytest.fixture
def ocp_dataset_handler_base():
    ocp_checkpoint_path = "/network/scratch/a/alexandre.duval/ocp/runs/4648581/checkpoints/best_checkpoint.pt"
    dataset_path = "/network/scratch/a/alexandre.duval/ocp/runs/4657270/deup_dataset"
    env = CrystalSurfaceEnv()
    try:
        dataset_handler = OCPDatasetHandler(
            env,
            ocp_checkpoint_path,
            dataset_path,
            float_precision=32,
        )
        return dataset_handler
    except AssertionError:
        pytest.skip(
            "folder not found. make sure that you have access to '/networks/scratch/a/alexandre.duval/ocp/'"
        )


@pytest.fixture
def ocp_dataset_handler_train_split():
    ocp_checkpoint_path = "/network/scratch/a/alexandre.duval/ocp/runs/4648581/checkpoints/best_checkpoint.pt"
    dataset_path = "/network/scratch/a/alexandre.duval/ocp/runs/4657270/deup_dataset"
    env = CrystalSurfaceEnv()
    try:
        dataset_handler = OCPDatasetHandler(
            env,
            ocp_checkpoint_path,
            dataset_path,
            train_fraction=0.1,
            float_precision=32,  # use train_fraction to create a train-test split from the training dataset
        )
        return dataset_handler
    except AssertionError:
        pytest.skip(
            "folder not found. make sure that you have access to '/networks/scratch/a/alexandre.duval/ocp/'"
        )


@pytest.mark.parametrize(
    "ocp_dataset_handler, train_shape, test_shape",
    [
        (
            "ocp_dataset_handler_base",
            torch.Size([40593, 352]),
            torch.Size([27017, 352]),
        ),
        (
            "ocp_dataset_handler_train_split",
            torch.Size([4059, 352]),
            torch.Size([36534, 352]),
        ),
    ],
)
def test__ocp_data__init(ocp_dataset_handler, train_shape, test_shape, request):
    ocp_dataset_handler = request.getfixturevalue(ocp_dataset_handler)
    assert ocp_dataset_handler.train_data.shape == train_shape
    assert ocp_dataset_handler.test_data.shape == test_shape


@pytest.mark.parametrize(
    "ocp_dataset_handler",
    [
        "ocp_dataset_handler_base",
        "ocp_dataset_handler_train_split",
    ],
)
def test__ocp_dataloader(ocp_dataset_handler, request):
    ocp_dataset_handler = request.getfixturevalue(ocp_dataset_handler)
    train_dataloader, test_dataloader = ocp_dataset_handler.get_dataloader()
    next(iter(train_dataloader))
    next(iter(test_dataloader))
    assert True


@pytest.mark.parametrize(
    "ocp_dataset_handler",
    [
        "ocp_dataset_handler_base",
        "ocp_dataset_handler_train_split",
    ],
)
def test__ocp_candidate_set(ocp_dataset_handler, request):
    ocp_dataset_handler = request.getfixturevalue(ocp_dataset_handler)
    candidate_set, _, _ = ocp_dataset_handler.get_candidate_set(return_index=True)
    next(iter(candidate_set))
    assert True
