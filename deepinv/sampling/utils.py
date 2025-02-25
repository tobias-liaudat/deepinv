import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from deepinv.utils.plotting import config_matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class Welford:
    r"""
    Welford's algorithm for calculating mean and variance

    https://doi.org/10.2307/1266577
    """

    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=lower, max=upper)


class CoveragePlotGenerator(nn.Module):

    def __init__(
        self,
        physics,
        dataset,
        sampling_model,
        confidence_levels,
        number_mc_samples=100,
        coverage_statistic="l2",  # HPD or l2-ball
        # convergence_criteria,
        device=torch.device("cpu"),
        verbose=False,
    ):
        super(CoveragePlotGenerator, self).__init__()

        # Init parameters here

        # Store inputs
        self.physics = physics.to(device)
        self.dataset = dataset
        self.sampling_model = sampling_model
        self.coverage_statistic = coverage_statistic
        self.confidence_levels = confidence_levels
        self.number_mc_samples = number_mc_samples
        # self.convergence_criteria = convergence_criteria
        self.device = device

        #
        self.dataset_len = len(dataset)

        # Build store tensors
        self.empirical_coverage = np.zeros((len(self.confidence_levels)))

    @staticmethod
    def mse(a, b):
        return (a - b).pow(2).reshape(a.shape[0], -1).mean(dim=1).detach().cpu().numpy()

    def coverage_plot(self, empirical_coverage=None, save_dir=None, show=True):
        config_matplotlib()

        if empirical_coverage is None:
            empirical_coverage = self.empirical_coverage

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        plt.plot(self.confidence_levels, self.empirical_coverage)
        plt.plot(self.confidence_levels, self.confidence_levels, "--", color="red")
        plt.title("Coverage Plot")

        if save_dir:
            plt.savefig(save_dir / "coverage_plots.png")
        if show:
            plt.show()

    def compute_coverage(self, batch_size=32):
        # if torch.cuda.is_available():
        #     device = dinv.utils.get_freer_gpu()
        # else:
        #     device = torch.device('cpu')

        if self.coverage_statistic == "l2":
            dataloader = DataLoader(self.dataset, batch_size, shuffle=False)
            for x, _ in dataloader:
                x = x.to(self.device)
                # print("x.shape: ", x.shape)
                y = self.physics(x)
                # print("y.shape: ", y.shape)
                mean, _ = self.sampling_model(
                    y, self.physics, x_init=self.physics.A_adj(x)
                )
                samples = self.sampling_model.get_chain()

                # print("len(samples): ", len(samples))
                # print("samples[0].shape: ", samples[0].shape)
                # print("mean.shape: ", mean.shape)

                true_mse = self.mse(x, mean)
                true_mse = torch.tensor(true_mse)

                estimated_mse = np.array([self.mse(sample, mean) for sample in samples])

                # print("len(estimated_mse): ", len(estimated_mse))
                # print("estimated_mse[0].shape: ", estimated_mse[0].shape)
                estimated_mse = torch.tensor(estimated_mse)

                for idx in range(len(self.confidence_levels)):
                    q = torch.quantile(
                        estimated_mse, self.confidence_levels[idx], dim=0
                    )
                    self.empirical_coverage[idx] += torch.sum(true_mse < q).cpu().item()

                # Reset markov chain model
                self.sampling_model.reset()

            self.empirical_coverage = self.empirical_coverage / self.dataset_len

            self.coverage_plot()
            return self.empirical_coverage
