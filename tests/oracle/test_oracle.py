import pytest
import torch

from activelearning.oracle.ocp import OCPOracle
from activelearning.oracle.oracle import BraninOracle, HartmannOracle


@pytest.fixture
def oracle_branin():
    return BraninOracle(
        fidelity=1, do_domain_map=True, device="cpu", float_precision=torch.float16
    )


@pytest.fixture
def oracle_hartmann():
    return HartmannOracle(fidelity=1, device="cpu", float_precision=torch.float16)


@pytest.fixture
def oracle_ocp():
    ocp_checkpoint_path = "/network/scratch/a/alexandre.duval/ocp/runs/4648581/checkpoints/best_checkpoint.pt"
    return OCPOracle(ocp_checkpoint_path, device="cpu", float_precision=torch.float16)


@pytest.mark.parametrize(
    "oracle",
    [
        "oracle_branin",
        "oracle_hartmann",
        "oracle_ocp",
    ],
)
def test__oracle__initializes_properly(oracle, request):
    oracle = request.getfixturevalue(oracle)
    assert True


@pytest.mark.parametrize(
    "oracle, input, output",
    [
        (
            "oracle_branin",
            torch.Tensor([[-1.0, -1.0], [-1.0, -0.1], [-0.1, -1.0], [0.8, 0.8]]),
            torch.Tensor(
                [
                    0.870941162109375,
                    187.3377685546875,
                    287.6764221191406,
                    168.0172119140625,
                ]
            ),
        ),
        (
            "oracle_hartmann",
            torch.Tensor(
                [
                    [-0.3333, -0.1111, -0.7778, 1.0000, -1.0000, -0.5556],
                    [-0.1111, -0.5556, -1.0000, 0.3333, 0.7778, -0.5556],
                    [0.5556, -0.5556, -1.0000, -0.1111, 0.1111, 1.0000],
                    [0.1111, 0.3333, -0.7778, -1.0000, -0.3333, 0.7778],
                ]
            ),
            torch.Tensor(
                [
                    1.0213609125120371e-10,
                    2.447236857605617e-09,
                    0.0007715617539361119,
                    1.5964351368635832e-11,
                ]
            ),
        ),
        # ()"oracle_ocp",) # TODO: test ocp oracle
    ],
)
def test__oracle__output(oracle, input, output, request):
    test_oracle = request.getfixturevalue(oracle)
    test_output = test_oracle(input)
    assert torch.equal(test_output, output)
