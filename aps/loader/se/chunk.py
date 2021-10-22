#!/usr/bin/env python

# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Dataloader of the raw waveform in enhancement/separation tasks
"""
import random
import numpy as np
import torch.utils.data as dat
import aps.distributed as dist

from torch.utils.data.dataloader import default_collate
from kaldi_python_io import Reader as BaseReader
from typing import List, Dict, Iterator, NoReturn, Union, Iterable
from aps.io.audio import AudioReader
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("se@chunk")
def DataLoader(train: bool = True,
               sr: int = 16000,
               mix_scp: str = "",
               doa_scp: str = "",
               ref_scp: str = "",
               emb_scp: str = "",
               chunk_size: int = 64000,
               max_batch_size: int = 16,
               distributed: bool = False,
               num_workers: int = 4) -> Iterable[Dict]:
    """
    Return a audio chunk dataloader for enhancement/separation tasks. We do audio chunking on the fly.
    Args:
        train: in training mode or not
        sr: sample rate of the audio
        mix_scp: mixture audio script, e.g., "mix.scp"
        emb_scp: speaker embedding script, e.g, "emb.scp" or ""
        doa_scp: DoA scripts, e.g., "spk1.scp" or "spk1.scp,spk2.scp" or ""
        ref_scp: reference audio scripts, e.g., "spk1.scp" or "spk1.scp,spk2.scp"
        chunk_size: #chunk_size (s)
        max_batch_size: #batch_size
        distributed: in distributed mode or not
        num_workers: number of workers used in dataloader
    """
    if not mix_scp:
        raise RuntimeError("mix_scp can not be None")

    def parse_args(scp_str):
        if not scp_str:
            return scp_str
        else:
            token = scp_str.split(",")
            return token[0] if len(token) == 1 else list(token)

    doa_scp = parse_args(doa_scp)
    ref_scp = parse_args(ref_scp)

    dataset = ScriptDataset(sr=sr,
                            mix_scp=mix_scp,
                            emb_scp=emb_scp,
                            doa_scp=doa_scp,
                            ref_scp=ref_scp)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=max_batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


class NumpyReader(BaseReader):
    """
    Sequential/Random Reader for numpy's ndarray (*.npy) file
    Args:
        npy_scp: script of numpy objects
    """

    def __init__(self, npy_scp: str) -> None:
        super(NumpyReader, self).__init__(npy_scp)

    def _load(self, key: str) -> np.ndarray:
        return np.load(self.index_dict[key])


class ScriptDataset(dat.Dataset):
    """
    Dataset configured by scripts (.scp file)
    Args:
        train: in training mode or not
        sr: sample rate of the audio
        mix_scp: mixture audio script, e.g., "mix.scp"
        emb_scp: speaker embedding script, e.g, "emb.scp" or ""
        doa_scp: DoA scripts, e.g., "spk1.scp" or "spk1.scp,spk2.scp" or ""
        ref_scp: reference audio scripts, e.g., "spk1.scp" or "spk1.scp,spk2.scp"
    """

    def __init__(self,
                 mix_scp: str = "",
                 doa_scp: Union[str, List[str]] = "",
                 emb_scp: str = "",
                 ref_scp: Union[str, List[str]] = "",
                 sr: int = 16000) -> None:
        self.mix = AudioReader(mix_scp, sr=sr)
        if isinstance(ref_scp, list):
            self.ref = [AudioReader(ref, sr=sr) for ref in ref_scp]
            self.num_ref = len(ref_scp)
        elif ref_scp:
            self.ref = AudioReader(ref_scp, sr=sr)
            self.num_ref = 1
        else:
            self.ref = None
            self.num_ref = 0
        self.num_doa = 0
        if isinstance(doa_scp, list):
            self.doa = [
                BaseReader(doa, value_processor=lambda x: np.float32(x))
                for doa in doa_scp
            ]
            self.num_doa = len(doa_scp)
        elif not doa_scp:
            self.doa = None
        else:
            self.doa = BaseReader(doa_scp,
                                  value_processor=lambda x: np.float32(x))
            self.num_doa = 1

        self.emb = NumpyReader(emb_scp) if emb_scp else None

    def _make_ref(self, key: str) -> Union[np.ndarray, List[np.ndarray]]:
        return self.ref[key] if self.num_ref == 1 else [
            reader[key] for reader in self.ref
        ]

    def _make_doa(self, key: str) -> Union[float, List[float]]:
        return self.doa[key] if self.num_doa == 1 else [
            reader[key] for reader in self.doa
        ]

    def _idx(self, key: str) -> Dict:
        eg = {}
        if self.ref is not None:
            eg["ref"] = self._make_ref(key)
        if self.doa is not None:
            eg["doa"] = self._make_doa(key)
        if self.emb is not None:
            eg["emb"] = self.emb[key]
        return eg

    def __getitem__(self, index: int) -> Dict:
        key = self.mix.index_keys[index]
        eg = self._idx(key)
        eg["mix"] = self.mix[key]
        return eg

    def __len__(self) -> int:
        return len(self.mix)

    def __iter__(self) -> Iterator[Dict]:
        for key, mix in self.mix:
            eg = self._idx(key)
            eg["mix"] = mix
            yield eg


class ChunkSplitter(object):
    """
    The class to split utterance into small chunks
    Args:
        chunk_size: size of audio chunk, we will split the long utterances
                    to several fix-length chunks
        train: in training mode or not
        hop: hop size between the chunks in one utterance
    """

    def __init__(self,
                 chunk_size: int,
                 train: bool = True,
                 hop: int = 16000) -> None:
        self.chunk_size = chunk_size
        self.hop = hop
        self.train = train

    def _chunk(self, mat_or_seq: Union[np.ndarray, List[np.ndarray]],
               s: int) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(mat_or_seq, list):
            return [mat[s:s + self.chunk_size] for mat in mat_or_seq]
        else:
            return mat_or_seq[s:s + self.chunk_size]

    def pad(self, mat_or_seq: Union[np.ndarray, List[np.ndarray]],
            pad_width: int) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(mat_or_seq, list):
            return [
                np.pad(mat, (0, pad_width), "constant") for mat in mat_or_seq
            ]
        else:
            return np.pad(mat_or_seq, (0, pad_width), "constant")

    def _make_chunk(self, eg: Dict, s: int) -> List[Dict]:
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "mbf": ndarray (beamforming results),
            "ref": ndarray or [ndarray, ...]
            "doa": float or [float, ...]
        """
        chunk = {}
        # support for multi-channel
        chunk["mix"] = eg["mix"][..., s:s + self.chunk_size]
        if "ref" in eg:
            chunk["ref"] = self._chunk(eg["ref"], s)
        if "doa" in eg:
            chunk["doa"] = eg["doa"]
        if "emb" in eg:
            chunk["emb"] = eg["emb"]
        return chunk

    def split(self, eg: Dict) -> List[Dict]:
        N = eg["mix"].shape[-1]
        # too short, throw away
        if N < self.hop:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            chunk = {}
            P = self.chunk_size - N

            pad_width = ((0, 0), (0, P)) if eg["mix"].ndim == 2 else (0, P)
            chunk["mix"] = np.pad(eg["mix"], pad_width, "constant")
            if "ref" in eg:
                chunk["ref"] = self.pad(eg["ref"], P)
            if "doa" in eg:
                chunk["doa"] = eg["doa"]
            if "emb" in eg:
                chunk["emb"] = eg["emb"]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.hop) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.hop
        return chunks


class WaveChunkDataLoader(object):
    """
    The audio chunk dataloader for SE/SS tasks (do chunk splitting on-the-fly)
    Args:
        dataset: instance of the audio dataset
        num_workers: number of the workers used in dataloader
        chunk_size: #chunk_size (s)
        batch_size: #batch_size
        distributed: in distributed mode or not
        train: in training mode or not
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 num_workers: int = 4,
                 chunk_size: int = 64000,
                 batch_size: int = 16,
                 distributed: bool = False,
                 train: bool = True) -> None:
        self.dataset = dataset
        self.train = train
        self.batch_size = batch_size
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      hop=chunk_size // 2)
        if distributed:
            self.sampler = dat.DistributedSampler(
                dataset,
                shuffle=train,
                num_replicas=dist.world_size(),
                rank=dist.rank())
        else:
            self.sampler = None
        # just return batch of egs, support multiple workers
        # NOTE: batch_size is not the batch_size of the audio chunk
        self.eg_loader = dat.DataLoader(self.dataset,
                                        batch_size=min(batch_size, 64),
                                        num_workers=num_workers,
                                        sampler=self.sampler,
                                        shuffle=(train and
                                                 self.sampler is None),
                                        collate_fn=self._collate)

    def _collate(self, batch):
        chunk = []
        for eg in batch:
            # split bss egs into target separation egs
            if isinstance(eg, list):
                for bias_eg in eg:
                    c = self.splitter.split(bias_eg)
                    chunk += c
            else:
                chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list: List[Dict]):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            batch["#utt"] = self.batch_size
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __len__(self) -> int:
        return 0

    def set_epoch(self, epoch: int) -> NoReturn:
        if self.sampler:
            self.sampler.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict]:
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
