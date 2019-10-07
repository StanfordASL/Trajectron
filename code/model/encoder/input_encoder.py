

class InputEncoder(object):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def encode(self, scene, node_history, node_state, node_future):
        raise NotImplementedError
