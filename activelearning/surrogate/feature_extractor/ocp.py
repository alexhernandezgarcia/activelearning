import torch.nn as nn
import torch_geometric
from activelearning.surrogate.feature_extractor.mlp import MLP
from ocpmodels.common.utils import make_trainer_from_dir
from ocpmodels.common.gfn import FAENetWrapper
from ocpmodels.datasets.data_transforms import get_transforms
from torch_scatter import scatter


class FAENetFeatureExtractor(nn.Module):
    def __init__(
        self, checkpoint_path, device, n_hidden=[32, 64], n_output=2, float_precision=32
    ):
        super(FAENetFeatureExtractor, self).__init__()
        self.device = device
        # have to set the ROOT to a folder that contains "configs/models/faenet.yaml" and "configs/models/tasks/is2re.yaml"
        # abs_config_dir = os.path.abspath("config/")
        # ocp_utils.ROOT = Path(abs_config_dir + "/surrogate/faenet").resolve()
        # ocp_utils.ROOT = Path(get_original_cwd() + "/config/surrogate/faenet").resolve()

        trainer = make_trainer_from_dir(
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
        trainer.load_checkpoint(checkpoint_path)

        self.mlp = MLP(
            n_input=trainer.model.module.embed_block.lin.out_features,
            n_hidden=n_hidden,
            n_output=n_output,
            float_precision=float_precision,
        ).to(self.device)

        wrapper = FAENetWrapper(
            faenet=trainer.model,
            transform=get_transforms(trainer.config),
            frame_averaging=trainer.config.get("frame_averaging", ""),
            trainer_config=trainer.config,
        )
        wrapper.freeze()
        self.faenet = wrapper
        self.trainer = trainer

    @property
    def n_output(self):
        return self.mlp.n_output

    def get_features(self, x, **kwargs):
        # if we get a graph instance, we need to extract the hidden states with faenet; otherwise, x is assumed to contain hidden states
        if isinstance(x, torch_geometric.data.batch.Batch):
            hidden_states = self.faenet(x, retrieve_hidden=True)["hidden_state"]
            assert (
                len(x.batch) == hidden_states.shape[0]
            ), "The output of the hidden state must be in graph format. To use an already scattered hidden state, the following line must be changed."
            x = scatter(hidden_states, x.batch.to(self.device), dim=0, reduce="mean")
        return self.mlp(x)

    def forward(self, x, **kwargs):
        return self.get_features(x)
