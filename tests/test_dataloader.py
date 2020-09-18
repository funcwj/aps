#!/usr/bin/env python

# wujian@2020

import pytest
import torch as th

from aps.loader import support_loader


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_am_wav_loader(batch_size, num_workers):
    egs_dir = "data/dataloader/am"
    loader = support_loader(fmt="am_wav",
                            wav_scp=f"{egs_dir}/egs.wav.scp",
                            token=f"{egs_dir}/egs.token",
                            utt2dur=f"{egs_dir}/egs.utt2dur",
                            train=False,
                            sr=16000,
                            adapt_dur=10,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            min_batch_size=1)
    for egs in loader:
        for key in ["src_pad", "tgt_pad", "tgt_len", "src_len"]:
            assert key in egs
        assert egs["src_pad"].shape == th.Size(
            [batch_size, egs["src_len"][0].item()])
        assert egs["tgt_pad"].shape == th.Size(
            [batch_size, egs["tgt_len"].max().item()])


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_workers", [0, 2, 4])
def test_am_kaldi_loader(batch_size, num_workers):
    egs_dir = "data/dataloader/am"
    loader = support_loader(fmt="am_kaldi",
                            feats_scp=f"{egs_dir}/egs.fbank.scp",
                            token=f"{egs_dir}/egs.token",
                            utt2dur=f"{egs_dir}/egs.fbank.num_frames",
                            train=False,
                            adapt_dur=900,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            min_batch_size=1)
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
    egs_dir = "data/dataloader/ss"
    loader = support_loader(fmt="ss_chunk",
                            mix_scp=f"{egs_dir}/wav.1.scp",
                            ref_scp=f"{egs_dir}/wav.1.scp",
                            sr=16000,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            chunk_size=chunk_size)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert egs["ref"].shape == th.Size([batch_size, chunk_size])
    loader = support_loader(fmt="ss_chunk",
                            mix_scp=f"{egs_dir}/wav.1.scp",
                            ref_scp=f"{egs_dir}/wav.1.scp,{egs_dir}/wav.1.scp",
                            sr=16000,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            chunk_size=chunk_size)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert len(egs["ref"]) == 2
        assert egs["ref"][0].shape == th.Size([batch_size, chunk_size])


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("chunk_size", [32000, 64000])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_ss_online_loader(batch_size, chunk_size, num_workers):
    egs_dir = "data/dataloader/ss"
    loader = support_loader(fmt="ss_online",
                            simu_cfg=f"{egs_dir}/online.opts",
                            sr=16000,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            chunk_size=chunk_size)
    for egs in loader:
        assert egs["mix"].shape == th.Size([batch_size, chunk_size])
        assert len(egs["ref"]) == 2
        assert egs["ref"][0].shape == th.Size([batch_size, chunk_size])


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_lm_utt_loader(batch_size):
    egs_dir = "data/dataloader/lm"
    loader = support_loader(fmt="lm_utt",
                            sos=0,
                            eos=1,
                            token=f"{egs_dir}/test.utt.token",
                            batch_size=batch_size,
                            drop_last=True)
    for egs in loader:
        assert egs["src"].shape == egs["tgt"].shape
        assert egs["src"].shape[0] == batch_size
