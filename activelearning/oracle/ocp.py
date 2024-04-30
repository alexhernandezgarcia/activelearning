from activelearning.oracle.oracle import Oracle
from ocpmodels.common.gfn import FAENetWrapper
from ocpmodels.datasets.data_transforms import get_transforms
from activelearning.utils.ocp import load_ocp_trainer


class OCPOracle(Oracle):
    def __init__(
        self, checkpoint_path, cost=1, fidelity=1, device="cpu", float_precision=64
    ):
        Oracle.__init__(
            self,
            cost,
            device,
            float_precision,
        )
        self.trainer = load_ocp_trainer(checkpoint_path)
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
