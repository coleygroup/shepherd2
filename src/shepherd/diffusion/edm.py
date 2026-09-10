"""
Module contains code for the EDM diffusion noise schedule.
"""

from dataclasses import dataclass, field
import numpy as np

import torch

@dataclass
class EDMNoiseScheduleOutput:
    sigma: np.ndarray = field(metadata={"help": "Noise level"})
    c_skip: np.ndarray = field(metadata={"help": "Skip scaling"})
    c_out: np.ndarray = field(metadata={"help": "Output scaling"})
    c_in: np.ndarray = field(metadata={"help": "Input scaling"})
    c_noise: np.ndarray = field(metadata={"help": "Noise condition AF3 style"})
    c_noise_edm: np.ndarray = field(metadata={"help": "Noise condition EDM style"})


class EDMNoiseSchedule:
    def __init__(
        self,
        sigma_data,
        P_mean = -1.2,
        P_std = 1.2,
        sigma_min = 1e-3,
        sigma_max = 80,
        rho = 7,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_scaling_factors(self, sigma: float | np.ndarray) -> EDMNoiseScheduleOutput:
        if isinstance(sigma, float):
            sigma = np.array([sigma])
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / np.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / np.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = np.log(sigma / self.sigma_data) / 4.
        c_noise_edm = np.log(sigma) / 4.

        return EDMNoiseScheduleOutput(
            sigma=sigma,
            c_skip=c_skip,
            c_out=c_out,
            c_in=c_in,
            c_noise_edm=c_noise_edm,
            c_noise=c_noise,
        )

    def sample_sigma(self, num_samples: int) -> np.ndarray:
        rand_normal = np.random.randn(num_samples)
        sigma = np.exp(self.P_mean + self.P_std * rand_normal)
        return sigma

    def get_sigma(self, t: np.ndarray) -> np.ndarray:
        # sigma = self.sigma_data * (self.sigma_max ** (1/self.rho) + t * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))**self.rho
        sigma = (self.sigma_max ** (1/self.rho) + t * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho)))**self.rho
        return sigma

    def __call__(self, t, training=False) -> EDMNoiseScheduleOutput:
        if training:
            sigma = self.sample_sigma(t.shape[0])
        else:
            sigma = self.get_sigma(t)
        return self.get_scaling_factors(sigma)

    def get_loss_weight(self, sigma: np.ndarray) -> np.ndarray:
        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        return loss_weight

    def get_noise(
        self,
        sigma: float,
        shape: tuple,
        remove_COM_from_noise: bool = False,
        mask: np.ndarray | None = None,
        scale_by_sigma_data: bool = False) -> np.ndarray:
        """
        Gets noise for a given noise level, sigma, and shape.

        Arguments
        ---------
        sigma: float of noise level.
        shape: tuple of shape of noise.
        remove_COM_from_noise: boolean flag to remove COM from noise.
        mask: boolean mask where virtual node is True and non-virtual node is False.

        Returns
        -------
        (N, D) array of noise
            noise * sigma if scale_by_sigma_data is True
            noise if scale_by_sigma_data is False
        """
        noise = np.random.randn(*shape) * sigma
        if scale_by_sigma_data:
            noise = noise * self.sigma_data # scale by sigma_data to match scale of data
        if remove_COM_from_noise:
            if mask is not None:
                noise = noise - noise[~mask].mean(0)
            else:
                noise = noise - noise.mean(0)
        if mask is not None:
            noise[mask, ...] = 0.0
        return noise

    def forward_noise(
        self,
        clean_input: np.ndarray,
        sigma: float,
        remove_COM_from_noise: bool = False,
        mask: np.ndarray | None = None) -> np.ndarray:
        """
        Given a clean data sample, it adds noise to the data sample for a given noise level, sigma.

        Arguments
        ---------
        clean_input: (N, D) array of input features.
        sigma: float of noise level.
        c_in: float of input scaling factor.
        remove_COM_from_noise: boolean flag to remove COM from noise.
        mask: boolean mask where virtual node is False and non-virtual node is True.
        scale_by_sigma_data: boolean flag to scale output by sigma_data.
            Helpful for models that expect input features to be in the same units as the data.

        Returns
        -------
        (N, D) array of forward-noised features
            c_in * (inp + noise)
            optionally scaled by sigma_data if scale_by_sigma_data is True
        """
        noise = self.get_noise(sigma, clean_input.shape, remove_COM_from_noise, mask)
        return clean_input + noise

    def scale_network_input(self, inp: np.ndarray, c_in: float, scale_by_sigma_data: bool = True) -> np.ndarray:
        """
        Scales the input by the input scaling factor, c_in, and optionally by the sigma_data.

        Arguments
        ---------
        inp: (N, D) array of input features.
        c_in: float of input scaling factor.
        scale_by_sigma_data: boolean flag to scale input by sigma_data.

        Returns
        -------
        (N, D) array of scaled input features
            c_in * inp * sigma_data if scale_by_sigma_data is True
            c_in * inp if scale_by_sigma_data is False
        """
        if scale_by_sigma_data:
            return c_in * inp * self.sigma_data
        else:
            return c_in * inp

    def scale_network_output(self, inp: np.ndarray, c_out: float) -> np.ndarray:
        """
        Scales the output of the networkby the output scaling factor, c_out.

        Arguments
        ---------
        inp: (N, D) array of input features.
        c_out: float of output scaling factor.

        Returns
        -------
        (N, D) array of scaled output features
            c_out * inp
        """
        return c_out * inp


@dataclass
class EDMScalingFactors:
    """Scaling factors from EDM; all tensors shape (N,) or (N, 1) for broadcasting with (N, D)."""
    sigma: torch.Tensor = field(metadata={"help": "Noise level"})
    c_skip: torch.Tensor = field(metadata={"help": "Skip scaling"})
    c_out: torch.Tensor = field(metadata={"help": "Output scaling"})
    c_in: torch.Tensor = field(metadata={"help": "Input scaling"})
    c_noise: torch.Tensor = field(metadata={"help": "Noise condition AF3 style"})
    c_noise_edm: torch.Tensor = field(metadata={"help": "Noise condition EDM style"})

class EDMPreconditioner:
    def __init__(
        self,
        sigma_data: float = 3.0,
        dtype: torch.dtype = torch.float32,
        clip_loss_weighting_max: float | None = None,
    ):
        self.sigma_data = torch.tensor(sigma_data)
        self.clip_loss_weighting_max = clip_loss_weighting_max

    def get_scaling_factors(self, sigma: torch.Tensor, unsqueeze: bool = True) -> EDMScalingFactors:
        """sigma: (N,) noise levels. Returns factors (N, 1) for broadcast with (N, D)."""
        device = sigma.device
        dtype = sigma.dtype
        sigma_data = self.sigma_data.to(device=device, dtype=dtype)
        sigma = sigma.to(dtype=dtype)
        s2 = sigma.square()
        sd2 = sigma_data.square()
        s2_plus_sd2 = s2 + sd2
        c_skip = sd2 / s2_plus_sd2
        c_out = sigma * sigma_data / torch.sqrt(s2_plus_sd2)
        c_in = 1.0 / torch.sqrt(s2_plus_sd2)
        c_noise = torch.log(sigma / sigma_data) / 4.0
        c_noise_edm = torch.log(sigma) / 4.0
        return EDMScalingFactors(
            sigma=sigma.unsqueeze(-1) if unsqueeze else sigma,
            c_skip=c_skip.unsqueeze(-1) if unsqueeze else c_skip,
            c_out=c_out.unsqueeze(-1) if unsqueeze else c_out,
            c_in=c_in.unsqueeze(-1) if unsqueeze else c_in,
            c_noise_edm=c_noise_edm.unsqueeze(-1) if unsqueeze else c_noise_edm,
            c_noise=c_noise.unsqueeze(-1) if unsqueeze else c_noise,
        )

    def get_loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """sigma: (N,) noise levels. Returns loss weight (N,1 ) for broadcast with (N, D)."""
        device = sigma.device
        dtype = sigma.dtype
        sigma_data = self.sigma_data.to(device=device, dtype=dtype)
        sigma = sigma.to(dtype=dtype)
        s2 = sigma.square()
        sd2 = sigma_data.square()
        loss_weight = (s2 + sd2) / (s2 * sd2)
        if self.clip_loss_weighting_max is not None:
            loss_weight = torch.clamp(loss_weight, max=self.clip_loss_weighting_max)
        return loss_weight.unsqueeze(-1)
