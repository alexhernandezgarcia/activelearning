from abc import ABC, abstractmethod
from gflownet.utils.common import set_float_precision
from typing import Union
import torch


class Oracle(ABC):
    def __init__(
        self,
        cost: float,
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
    ):
        self.cost = cost
        self.device = device
        self.float_precision = set_float_precision(float_precision)

    @abstractmethod
    def __call__(self, states) -> torch.Tensor:
        pass


try:
    from gflownet.proxy.box.branin import Branin

    class BraninOracle(Oracle, Branin):
        def __init__(
            self,
            cost=1,
            fidelity=1,
            do_domain_map=True,
            negate=False,
            device="cpu",
            float_precision=64,
        ):
            Oracle.__init__(
                self,
                cost,
                device,
                float_precision,
            )
            Branin.__init__(
                self,
                fidelity=fidelity,
                do_domain_map=do_domain_map,
                device=self.device,
                float_precision=self.float_precision,
                negate=negate,
            )

        def __call__(self, states):
            return Branin.proxy2reward(self, Branin.__call__(self, states.clone()))

except ImportError:
    print("please install gflownet to use the branin proxy")


try:
    from gflownet.proxy.box.hartmann import Hartmann

    class HartmannOracle(Oracle, Hartmann):
        def __init__(
            self,
            cost=1,
            fidelity=1,
            device="cpu",
            float_precision=64,
            negate=False,
        ):
            Oracle.__init__(
                self,
                cost,
                device,
                float_precision,
            )
            Hartmann.__init__(
                self,
                fidelity=fidelity,
                device=self.device,
                float_precision=self.float_precision,
                negate=negate,
            )

        def __call__(self, states):
            return Hartmann.proxy2reward(self, Hartmann.__call__(self, states.clone()))

except ImportError:
    print("please install gflownet to use the hartmann proxy")


# class MultiFidelityOracle(Oracle):
#     def __init__(self):
#         super().__init__()
