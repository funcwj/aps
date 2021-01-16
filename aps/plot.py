# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import numpy as np
import matplotlib.pyplot as plt

from typing import NoReturn, Union

default_font = "Times New Roman"
default_dpi = 200
default_fmt = "jpg"


def plot_feature(feat: Union[np.ndarray, th.Tensor],
                 dest: str,
                 cmap: str = "jet",
                 hop: int = 256,
                 sr: int = 16000,
                 aspect: str = "auto",
                 interpolation: str = "antialiased") -> NoReturn:
    """
    Plot acoustic feature
    Args:
        feat (ndarray, Tensor): T x F, acoustic features
        dest (str): path to save the figure
        cmap (str): color map
        hop (int): frame hop size
        sr (int): sample rate of the original audio
    """
    if isinstance(feat, th.Tensor):
        feat = feat.numpy()
    assert feat.ndim == 2
    num_frames, num_bins = feat.shape
    fig, axis = plt.subplots()
    axis.imshow(feat.T,
                origin="lower",
                cmap=cmap,
                aspect=aspect,
                interpolation=interpolation)
    # time axis
    xticks = np.linspace(0, num_frames - 1, 5)
    xlabels = [f"{t:.2f}" for t in (xticks * hop / sr)]
    axis.set_xticks(xticks)
    axis.set_xticklabels(xlabels, fontproperties=default_font)
    axis.set_xlabel("Time (s)", fontdict={"family": default_font})
    # frequency axis
    yticks = np.linspace(0, num_bins - 1, 6)
    ylabels = [f"{t / 1000:.1f}" for t in np.linspace(0, sr / 2, 6)]
    axis.set_yticks(yticks)
    axis.set_yticklabels(ylabels, fontproperties=default_font)
    axis.set_ylabel("Frequency (kHz)", fontdict={"family": default_font})
    # save figure
    fig.savefig(f"{dest}.{default_fmt}", dpi=default_dpi, format=default_fmt)
    plt.close(fig)
