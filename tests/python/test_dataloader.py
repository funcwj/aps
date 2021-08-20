#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th

from aps.libs import aps_dataloader
from aps.conf import load_dict


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_am_raw_loader(batch_size, num_workers):
    egs_dir = "data/dataloader/am"
    loader = aps_dataloader(fmt="am@raw",
                            wav_scp=f"{egs_dir}/egs.wav.scp",
                            text=f"{egs_dir}/egs.fake.text",
                            utt2dur=f"{egs_dir}/egs.utt2dur",
                            vocab_dict=load_dict(f"{egs_dir}/dict"),
                            train=False,
                            sr=16000,
                            adapt_dur=10,
                            num_workers=num_workers,
                            max_batch_size=batch_size,
                            min_batch_size=1)
    for egs in loader:
        for key in ["src_pad", "tgt_pad", "tgt_len", "src_len"]:
            assert key in egs
        assert egs["src_pad"].shape == th.Size(
            [batch_size, egs["src_len"][0].item()])
        assert egs["tgt_pad"].shape == th.Size(
            [batch_size, egs["tgt_len"].max().item()])


@pytest.mark.parametrize("batch_size", [10, 15])
@pytest.mark.parametrize("num_workers", [2, 4])
def test_am_raw_loader_const(batch_size, num_workers):
    egs_dir = "data/dataloader/am"
    loader = aps_dataloader(fmt="am@raw",
                            wav_scp=f"{egs_dir}/egs.wav.scp",
                            text=f"{egs_dir}/egs.fake.text",
                            utt2dur=f"{egs_dir}/egs.utt2dur",
                            vocab_dict=load_dict(f"{egs_dir}/dict"),
                            train=False,
                            sr=16000,
                            max_batch_size=batch_size,
                            batch_mode="constraint",
                            num_workers=num_workers)
    for egs in loader:
        for key in ["src_pad", "tgt_pad", "tgt_len", "src_len"]:
            assert key in egs
        assert egs["tgt_pad"].shape[-1] == egs["tgt_len"].max().item()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_am_kaldi_loader(batch_size, num_workers):
    egs_dir = "data/dataloader/am"
    loader = aps_dataloader(fmt="am@kaldi",
                            feats_scp=f"{egs_dir}/egs.fbank.scp",
                            text=f"{egs_dir}/egs.fake.text",
                            vocab_dict=load_dict(f"{egs_dir}/dict"),
                            utt2num_frames=f"{egs_dir}/egs.fbank.num_frames",
                            train=False,
                            adapt_dur=900,
                            num_workers=num_workers,
                            min_batch_size=1,
                            max_batch_size=batch_size)
    for egs in loader:
        for key in ["src_pad", "tgt_pad", "tgt_len", "src_len"]:
            assert key in egs

        assert egs["src_pad"].shape == th.Size(
            [batch_size, egs["src_len"][0].item(), 80])
        assert egs["tgt_pad"].shape == th.Size(
            [batch_size, egs["tgt_len"].max().item()])


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [32000, 64000])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_ss_chunk_loader(batch_size, chunk_size, num_workers):
    egs_dir = "data/dataloader/se"
    loader = aps_dataloader(fmt="se@chunk",
                            mix_scp=f"{egs_dir}/wav.1.scp",
                            ref_scp=f"{egs_dir}/wav.1.scp",
                            sr=16000,
                            max_batch_size=batch_size,
                            chunk_size=chunk_size,
                            num_workers=num_workers)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert egs["ref"].shape == th.Size([batch_size, chunk_size])
    loader = aps_dataloader(fmt="se@chunk",
                            mix_scp=f"{egs_dir}/wav.1.scp",
                            ref_scp=f"{egs_dir}/wav.1.scp,{egs_dir}/wav.1.scp",
                            sr=16000,
                            max_batch_size=batch_size,
                            chunk_size=chunk_size,
                            num_workers=num_workers)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert len(egs["ref"]) == 2
        assert egs["ref"][0].shape == th.Size([batch_size, chunk_size])


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [32000, 64000])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_ss_command_loader(batch_size, chunk_size, num_workers):
    egs_dir = "data/dataloader/se"
    loader = aps_dataloader(fmt="se@command",
                            simu_cfg=f"{egs_dir}/cmd.options",
                            sr=16000,
                            max_batch_size=batch_size,
                            chunk_size=chunk_size,
                            num_workers=num_workers)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert len(egs["ref"]) == 2
        assert egs["ref"][0].shape == th.Size([batch_size, chunk_size])


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("obj", ["egs.token", "egs.token.gz"])
def test_lm_utt_loader(batch_size, obj):
    egs_dir = "data/dataloader/lm"
    loader = aps_dataloader(fmt="lm@utt",
                            sos=1,
                            eos=2,
                            text=f"{egs_dir}/{obj}",
                            vocab_dict=load_dict(f"{egs_dir}/dict"),
                            max_batch_size=batch_size,
                            min_batch_size=batch_size)
    for egs in loader:
        assert egs["src"].shape == egs["tgt"].shape
        assert egs["src"].shape[0] == batch_size


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("obj", ["egs.token", "egs.token.gz"])
def test_lm_bptt_loader(batch_size, obj):
    egs_dir = "data/dataloader/lm"
    loader = aps_dataloader(fmt="lm@bptt",
                            sos=1,
                            eos=2,
                            text=f"{egs_dir}/{obj}",
                            vocab_dict=load_dict(f"{egs_dir}/dict"),
                            bptt_size=10,
                            max_batch_size=batch_size)
    for egs in loader:
        assert egs["src"].shape == egs["tgt"].shape
        assert egs["src"].shape == th.Size([batch_size, 10])
