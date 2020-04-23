#!/usr/bin/env python

# wujian@2020
"""
Dataloader for raw waveforms in enhancement/separation tasks
"""
import random
import torch as th
import numpy as np
import torch.utils.data as dat

from torch.utils.data.dataloader import default_collate
from kaldi_python_io import Reader as BaseReader

from ..am.wav import WaveReader

type_seq = (list, tuple)


def DataLoader(train=True,
               sr=16000,
               mix_scp="",
               doa_scp="",
               ref_scp="",
               emb_scp="",
               chunk_size=64000,
               batch_size=16,
               num_workers=4):
    """
    Return a online-chunk dataloader for enhancement/separation tasks
    args
        mix_scp: "mix.scp"
        emb_scp: "emb.scp" or ""
        doa_scp: "spk1.scp" or "spk1.scp,spk2.scp" or ""
        ref_scp: "spk1.scp" or "spk1.scp,spk2.scp"
    """
    if not mix_scp:
        raise RuntimeError("mix_scp can not be None")

    def parse_args(scp_str):
        if not scp_str:
            return scp_str
        else:
            token = scp_str.split(",")
            return token[0] if len(token) == 1 else token

    doa_scp = parse_args(doa_scp)
    ref_scp = parse_args(ref_scp)

    dataset = ScriptDataset(shuffle=train,
                            sr=sr,
                            mix_scp=mix_scp,
                            emb_scp=emb_scp,
                            doa_scp=doa_scp,
                            ref_scp=ref_scp)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=batch_size,
                               num_workers=num_workers)


class NumpyReader(BaseReader):
    """
    Sequential/Random Reader for numpy's ndarray(*.npy) file
    """
    def __init__(self, npy_scp):
        super(NumpyReader, self).__init__(npy_scp)

    def _load(self, key):
        return np.load(self.index_dict[key])


class ScriptDataset(object):
    """
    Dataset configured by scripts
    """
    def __init__(self,
                 shuffle=False,
                 mix_scp="",
                 doa_scp="",
                 emb_scp="",
                 ref_scp=None,
                 sr=16000):
        self.mix = WaveReader(mix_scp, sr=sr)
        if isinstance(ref_scp, type_seq):
            self.ref = [WaveReader(ref, sr=sr) for ref in ref_scp]
            self.num_ref = len(ref_scp)
        elif ref_scp:
            self.ref = WaveReader(ref_scp, sr=sr)
            self.num_ref = 1
        else:
            self.ref = None
            self.num_ref = 0
        self.num_doa = 0
        if isinstance(doa_scp, type_seq):
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

        self.shuffle = shuffle

    def _make_ref(self, key):
        return self.ref[key] if self.num_ref == 1 else [
            reader[key] for reader in self.ref
        ]

    def _make_doa(self, key):
        return self.doa[key] if self.num_doa == 1 else [
            reader[key] for reader in self.doa
        ]

    def _idx(self, key):
        eg = {}
        if self.ref is not None:
            eg["ref"] = self._make_ref(key)
        if self.doa is not None:
            eg["doa"] = self._make_doa(key)
        if self.emb is not None:
            eg["emb"] = self.emb[key]
        return eg

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        eg = self._idx(key)
        eg["mix"] = self.mix[key]
        return eg

    def __len__(self):
        return len(self.mix)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.mix.index_keys)
        for key, mix in self.mix:
            eg = self._idx(key)
            eg["mix"] = mix
            yield eg


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, hop=16000):
        self.chunk_size = chunk_size
        self.hop = hop
        self.train = train

    def _chunk(self, mat_or_seq, s):
        if isinstance(mat_or_seq, type_seq):
            return [mat[s:s + self.chunk_size] for mat in mat_or_seq]
        else:
            return mat_or_seq[s:s + self.chunk_size]

    def pad(self, mat_or_seq, pad_width):
        if isinstance(mat_or_seq, type_seq):
            return [
                np.pad(mat, (0, pad_width), "constant") for mat in mat_or_seq
            ]
        else:
            return np.pad(mat_or_seq, (0, pad_width), "constant")

    def _make_chunk(self, eg, s):
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

    def split(self, eg):
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
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=64000,
                 batch_size=16,
                 train=True):
        self.dataset = dataset
        self.train = train
        self.batch_size = batch_size
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      hop=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(self.dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        chunk = []
        for eg in batch:
            # split bss egs into target separation egs
            if isinstance(eg, type_seq):
                for bias_eg in eg:
                    c = self.splitter.split(bias_eg)
                    chunk += c
            else:
                chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __len__(self):
        return 0

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj