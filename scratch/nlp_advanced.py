from scratch.deep_learning import Optimizer, Layer

class EmbeddingOptimizer(Optimizer):
    """
    Optimized for the case where there are
    only embedding layers with single id updates.
    """
    def __init__(self, learning_rate: float) -> None:
        self.lr = learning_rate

    def step(self, layer: Layer) -> None:
        for param, grad in zip(layer.params(), layer.grads()):
            # Find the first (only) row with nonzero values.
            for idx, row in enumerate(grad):
                if row[0] != 0:
                    break

            # Then update just that row.
            for j in range(len(row)):
                param[idx][j] -= grad[idx][j] * self.lr
