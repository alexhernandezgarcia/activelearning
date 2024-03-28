from abc import ABC, abstractmethod
from gflownet.proxy.box.branin import Branin
from gflownet.proxy.box.hartmann import Hartmann
from gflownet.utils.common import set_float_precision


class Oracle(ABC):
    def __init__(self, cost, device, float_precision):
        self.cost = cost
        self.device = device
        self.float_precision = set_float_precision(float_precision)

    @abstractmethod
    def __call__(self):
        pass


class BraninOracle(Oracle, Branin):
    def __init__(
        self, cost=1, fidelity=1, do_domain_map=True, device="cpu", float_precision=64
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
        )

    def __call__(self, states):
        return Branin.__call__(self, states)


class HartmannOracle(Oracle, Hartmann):
    def __init__(self, cost=1, fidelity=1, device="cpu", float_precision=64):
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
        )

    def __call__(self, states):
        return Hartmann.__call__(self, states)


# class MultiFidelityOracle(Oracle):
#     def __init__(self):
#         super().__init__()
