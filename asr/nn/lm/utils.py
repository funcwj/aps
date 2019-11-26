# wujian@2019

import codecs


class NbestReader(object):
    """
    Read nbest hypos
    """
    def __init__(self, obj):
        self.nbest = 1
        self.hypos = {}
        with codecs.open(obj, "r", encoding="utf-8") as f:
            self.nbest = int(f.readline())
            while True:
                key = f.readline().strip()
                if not key:
                    break
                topk = []
                for _ in range(self.nbest):
                    items = f.readline().strip().split()
                    score = float(items[0])
                    trans = " ".join(items[1:])
                    topk.append((score, trans))
                self.hypos[key] = topk

    def __len__(self):
        return len(self.hypos)

    def __getitem__(self, key):
        return self.hypos[key]

    def __iter__(self):
        for key in self.hypos:
            yield key, self.hypos[key]