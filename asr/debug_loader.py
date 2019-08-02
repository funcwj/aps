#!/usr/bin/env python

# wujian@2019

from libs.dataset import make_dataloader


def run():
    loader = make_dataloader(feats_scp="",
                             token_scp="",
                             utt2num_frames="",
                             train=True,
                             batch_size=32)
    for egs in loader:
        print(egs["x_pad"].shape)


if __name__ == "__main__":
    run()