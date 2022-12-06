from model.base import Model

class MLP(Model):
    def __init__(self, config, logger, init_model=False):
        super().__init__(config, logger)
        self.model_class = MLP
        self.device = config.device
        if init_model:
            self.init_model()

    def init_model(self):
        super().init_model()

    def load_model(self, dir_name=None):
        super().load_model(dir_name)

    def converge(self, data_handler):
        super().converge(data_handler)

    def train(self, tr):
        super().train(tr)

    def test(self, te):
        super().test(te)

    def get_loss(self, data):
        return super().get_loss(data)

    def check_convergence(self):
        super().check_convergence()

    def evaluate(self, data):
        return super().evaluate(data)