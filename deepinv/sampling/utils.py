import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from deepinv.utils.plotting import config_matplotlib
import matplotlib.pyplot as plt


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
        use_online_stats=False,
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
        self.use_online_stats = use_online_stats
        self.device = device
        self.dataset_len = len(dataset)
        self.verbose = verbose

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

        if self.coverage_statistic == "l2":
            dataloader = DataLoader(self.dataset, batch_size, shuffle=False)
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                mean, var = self.sampling_model(
                    y, self.physics, x_init=self.physics.A_adjoint(y)
                )

                # Get oracle error
                true_mse = self.mse(x, mean)
                true_mse = torch.tensor(true_mse)

                if self.use_online_stats:
                    # Shape estimated_mse = [number_samples, batch_size, len(statistic_online_func)]
                    estimated_mse = self.sampling_model.get_online_statistics()
                    # Turning the nested list of statistics into a np.ndarray and then to tensor
                    estimated_mse = np.array(estimated_mse)
                    estimated_mse = torch.tensor(estimated_mse[0])
                    # TODO: Add iteration over online statistic functions

                else:
                    samples = self.sampling_model.get_chain()
                    estimated_mse = np.array(
                        [self.mse(sample, mean) for sample in samples]
                    )
                    # Shape estimated_mse = [number_samples, batch_size]
                    estimated_mse = torch.tensor(estimated_mse)

                # TODO: Add iteration over online statistic functions
                for idx in range(len(self.confidence_levels)):
                    q = torch.quantile(
                        estimated_mse, self.confidence_levels[idx], dim=0
                    )
                    self.empirical_coverage[idx] += torch.sum(true_mse < q).cpu().item()

                # Reset markov chain model
                self.sampling_model.reset()

            self.empirical_coverage = self.empirical_coverage / self.dataset_len

            # TODO: Add iteration over online statistic functions in the plots
            self.coverage_plot()
            return self.empirical_coverage
