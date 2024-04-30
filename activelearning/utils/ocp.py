from pathlib import Path
from ocpmodels.common.utils import make_trainer_from_dir
from ocpmodels.common.gfn import FAENetWrapper
from ocpmodels.datasets.data_transforms import get_transforms
from ocpmodels.common import utils as ocp_utils
import os


def load_ocp_trainer(checkpoint_path):
    abs_config_dir = os.path.abspath("config/")
    # have to set the ROOT to a folder that contains "configs/models/faenet.yaml" and "configs/models/tasks/is2re.yaml"
    ocp_utils.ROOT = Path(abs_config_dir + "/surrogate/faenet").resolve()

    trainer = make_trainer_from_dir(
        checkpoint_path,
        mode="continue",
        overrides={
            "is_debug": True,
            "silent": True,
            "cp_data_to_tmpdir": False,
            "config": "faenet-deup_is2re-all",
            "deup_dataset.create": False,
            "model": {
                "dropout_lin": 0.0
            },  # for inference, we don't want to have dropout
        },
        skip_imports=["qm7x", "gemnet", "spherenet", "painn", "comenet"],
        silent=True,
    )
    return trainer
