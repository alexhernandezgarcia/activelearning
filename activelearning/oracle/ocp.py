from activelearning.oracle.oracle import Oracle
from ocpmodels.common.gfn import FAENetWrapper
from ocpmodels.datasets.data_transforms import get_transforms
from ocpmodels.common.utils import make_trainer_from_dir


class OCPOracle(Oracle):
    def __init__(self, checkpoint_path, cost=1, device="cpu", float_precision=64):
        Oracle.__init__(
            self,
            cost,
            device,
            float_precision,
        )

        # have to set the ROOT to a folder that contains "configs/models/faenet.yaml" and "configs/models/tasks/is2re.yaml"
        # abs_config_dir = os.path.abspath("config/")
        # ocp_utils.ROOT = Path(abs_config_dir + "/surrogate/faenet").resolve()
        # ocp_utils.ROOT = Path(get_original_cwd() + "/config/surrogate/faenet").resolve()

        self.trainer = make_trainer_from_dir(
            checkpoint_path,
            mode="continue",
            overrides={
                "is_debug": True,
                "silent": True,
                "cp_data_to_tmpdir": False,
                "deup_dataset.create": False,
                "model": {
                    "dropout_lin": 0.0
                },  # for inference, we don't want to have dropout
                "cpu": device == "cpu",
            },
            skip_imports=["qm7x", "gemnet", "spherenet", "painn", "comenet"],
            silent=True,
        )
        self.trainer.load_checkpoint(checkpoint_path)

        wrapper = FAENetWrapper(
            faenet=self.trainer.model,
            transform=get_transforms(self.trainer.config),
            frame_averaging=self.trainer.config.get("frame_averaging", ""),
            trainer_config=self.trainer.config,
        )
        wrapper.freeze()
        self.model = wrapper

    def __call__(self, states):
        states = states.clone()
        return self.model(states, retrieve_hidden=False)


# class GroundTruthOracle(OCPOracle):
#     def __call__(self, states):
