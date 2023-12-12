import math
import torch
import torch.nn.functional as F


def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, n_dims_truncated, n_points, batch_size):
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.b_size = batch_size
        self.n_points = n_points

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=None
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "noisy_linear_regression": NoisyLinearRegression,  # std=0.1
        "sparse_linear_regression": SparseLinearRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=sparsity, **args)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):

    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=None):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, n_dims_truncated, n_points, batch_size)
        self.device = device
        self.xs = torch.randn(batch_size, n_points, n_dims, device=device)  # [B, n, d]
        self.xs[..., n_dims_truncated:] = 0
        w_b = torch.randn(batch_size, n_dims, 1, device=device)  # [B, d, 1]
        w_b[:, n_dims_truncated:] = 0
        self.w_b = w_b
        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class NoisyLinearRegression(LinearRegression):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=None, std=0.1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(NoisyLinearRegression, self).__init__(batch_size, n_points, n_dims, n_dims_truncated, device)
        self.ys += torch.randn_like(self.ys) * std

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=3):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(batch_size, n_points, n_dims, n_dims_truncated, device)
        self.sparsity = sparsity
        valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):  # [B, d, 1]
            mask = torch.ones(n_dims).bool()  # [d]
            perm = torch.randperm(valid_coords)
            mask[perm[:sparsity]] = False
            w[mask] = 0  # w shape [d, 1]

        self.ys = (self.xs @ self.w_b).sum(-1)  # [B, n]

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class Relu2nnRegression(Task):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=100, hidden_layer_size=100):
        super(Relu2nnRegression, self).__init__(n_dims, n_dims_truncated, n_points, batch_size)
        self.hidden_layer_size = hidden_layer_size

        W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size, device=device)
        W2 = torch.randn(self.b_size, hidden_layer_size, 1, device=device)

        if sparsity < hidden_layer_size:
            import random
            non_sparse_mask = torch.zeros(hidden_layer_size, device=device)
            non_sparse_indices = random.sample(range(hidden_layer_size), sparsity)
            non_sparse_mask[non_sparse_indices] = 1
            self.W1 = W1 * non_sparse_mask[None, None, :]
            self.W2 = W2 * non_sparse_mask[None, :, None]
        else:
            self.W1 = W1
            self.W2 = W2

        self.xs = torch.randn(batch_size, n_points, n_dims, device=device)  # [B, n, d]
        self.xs[..., n_dims_truncated:] = 0

        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs_b):
        W1 = self.W1
        W2 = self.W2
        # Renormalize to Linear Regression Scale
        ys_b_nn = (F.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        # ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, batch_size, n_points, n_dims, n_dims_truncated, device, sparsity=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, n_dims_truncated, n_points, batch_size)
        self.depth = depth

        # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
        # dt_tensor stores the coordinate used at each node of the decision tree.
        # Only indices corresponding to non-leaf nodes are relevant
        self.dt_tensor = torch.randint(
            low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
        )

        # Target value at the leaf nodes.
        # Only indices corresponding to leaf nodes are relevant.
        self.target_tensor = torch.randn(self.dt_tensor.shape)

        self.xs = torch.randn(batch_size, n_points, n_dims, device=device)  # [B, n, d]
        self.ys = self.evaluate(self.xs)

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
