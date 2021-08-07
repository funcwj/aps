# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from aps.transform.utils import STFTBase
from aps.const import EPSILON
"""
For streaming version of the (i)STFT, we use rfft/irfft for efficient processing
"""


class StreamingSTFT(STFTBase):
    """
    To mimic streaming (frame by frame processing) STFT
    """

    def __init__(self, *args, **kwargs):
        super(StreamingSTFT, self).__init__(*args, inverse=False, **kwargs)

    def step(self, frame: th.Tensor, return_polar: bool = False) -> th.Tensor:
        """
        Process one frame
        Args:
            frame (Tensor): N x (C) x S
        Return:
            frame (Tensor): N x (C) x F x 2
        """
        frame = th.fft.rfft(frame * self.w,
                            self.win_length,
                            dim=-1,
                            norm="ortho" if self.normalized else "backward")
        # N x C x F x 2
        frame = th.view_as_real(frame)
        if return_polar:
            mag = (th.sum(frame**2, -1) + EPSILON)**0.5
            pha = th.atan2(frame[..., 1], frame[..., 0])
            frame = th.stack([mag, pha], dim=-1)
        return frame

    def forward(self, wav: th.Tensor, return_polar: bool = False) -> th.Tensor:
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        Args
            wav (Tensor): input signal, N x (C) x S
        Return
            transform (Tensor): N x (C) x F x T x 2
        """
        frames = []
        num_samples = wav.shape[-1]
        for t in range(0, num_samples - self.win_length + 1, self.frame_hop):
            frame = self.step(wav[..., t:t + self.win_length],
                              return_polar=return_polar)
            frames.append(frame)
        return th.stack(frames, -2)


class StreamingiSTFT(STFTBase):
    """
    To mimic streaming (frame by frame processing) iSTFT
    """

    def __init__(self, *args, **kwargs):
        super(StreamingiSTFT, self).__init__(*args, inverse=True, **kwargs)
        self.reset()

    def reset(self):
        self.wav_cache = th.zeros(self.win_length - self.frame_hop)
        self.win_cache = th.zeros_like(self.wav_cache)

    def step(self, frame: th.Tensor, return_polar: bool = False) -> th.Tensor:
        """
        Process one frame
        Args:
            frame (Tensor): N x F x 2
        Return
            frame (Tensor): N x W
        """
        if return_polar:
            real = frame[..., 0] * th.cos(frame[..., 1])
            imag = frame[..., 0] * th.sin(frame[..., 1])
            frame = th.stack([real, imag], -1)
        frame = th.view_as_complex(frame)
        frame = th.fft.irfft(frame,
                             self.win_length,
                             dim=-1,
                             norm="ortho" if self.normalized else "backward")
        return frame * self.w

    def norm(self, frame: th.Tensor) -> th.Tensor:
        """
        Normalize the frame
        """
        window = (self.w**2).clone()
        overlap_len = self.win_cache.shape[0]
        frame[:, :overlap_len] += self.wav_cache
        window[:overlap_len] += self.win_cache
        self.win_cache = window[self.frame_hop:]
        self.wav_cache = frame[:, self.frame_hop:]
        frame = frame / (window + EPSILON)
        return frame[:, :self.frame_hop]

    def forward(self,
                transform: th.Tensor,
                return_polar: bool = False) -> th.Tensor:
        """
        Accept phase & magnitude and output raw waveform
        Args
            transform (Tensor): STFT output, N x F x T x 2
        Return
            wav (Tensor): N x S
        """
        self.reset()
        frames = []
        num_frames = transform.shape[-2]
        for t in range(num_frames):
            frame = self.step(transform[..., t, :], return_polar=return_polar)
            frames.append(self.norm(frame))
        cache = self.wav_cache / (self.win_cache + EPSILON)
        wav = th.cat(frames + [cache], -1)
        return wav
