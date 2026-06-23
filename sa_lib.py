"""Suffix-array based popular sub-route mining + a minimal HyperLogLog. Shared by test & notebook."""
import numpy as np, hashlib
from math import radians, sin, cos, asin, sqrt
import h3


def suffix_array(s):
    """Prefix-doubling suffix array (numpy). s: 1D int array (ranks >=0)."""
    s = np.asarray(s, dtype=np.int64)
    n = len(s)
    _, rank = np.unique(s, return_inverse=True)
    rank = rank.astype(np.int64)
    k = 1
    while True:
        rank2 = np.full(n, -1, dtype=np.int64)
        rank2[:n - k] = rank[k:]
        order = np.lexsort((rank2, rank))
        rs, r2s = rank[order], rank2[order]
        diff = np.empty(n, dtype=bool)
        diff[0] = False
        diff[1:] = (rs[1:] != rs[:-1]) | (r2s[1:] != r2s[:-1])
        newrank = np.empty(n, dtype=np.int64)
        newrank[order] = np.cumsum(diff)
        rank = newrank
        if rank.max() == n - 1:
            break
        k *= 2
    return order  # the suffix array


def kasai_lcp(s, sa):
    """LCP array via Kasai. lcp[r] = lcp(sa[r-1], sa[r]); lcp[0]=0."""
    s = np.asarray(s, dtype=np.int64)
    n = len(s)
    rank = np.empty(n, dtype=np.int64)
    rank[sa] = np.arange(n)
    lcp = np.zeros(n, dtype=np.int64)
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp


class HLL:
    """Minimal HyperLogLog for distinct-count estimation (p bits -> m=2^p registers)."""
    def __init__(self, p=12):
        self.p = p
        self.m = 1 << p
        self.reg = np.zeros(self.m, dtype=np.int8)
        self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def add_int(self, x):
        h = int.from_bytes(hashlib.blake2b(int(x).to_bytes(8, "little"), digest_size=8).digest(), "little")
        idx = h & (self.m - 1)
        w = h >> self.p                       # remaining 64-p bits
        rho = (64 - self.p) - w.bit_length() + 1   # leading zeros + 1
        if rho > self.reg[idx]:
            self.reg[idx] = rho

    def count(self):
        m = self.m
        z = 1.0 / np.power(2.0, self.reg.astype(np.float64))
        E = self.alpha * m * m / z.sum()
        if E <= 2.5 * m:
            v = int((self.reg == 0).sum())
            if v > 0:
                E = m * np.log(m / v)
        return int(E)


def build_sequence(cell_lists):
    """Map cells->int ids, concat with unique per-trip sentinels. Returns seq, doc_id, id2cell, C."""
    vocab = {}
    for cells in cell_lists:
        for c in cells:
            if c not in vocab:
                vocab[c] = len(vocab)
    C = len(vocab)               # real symbols 0..C-1 ; sentinels >= C (unique per trip)
    seq, doc = [], []
    sep = C
    for t, cells in enumerate(cell_lists):
        for c in cells:
            seq.append(vocab[c]); doc.append(t)
        seq.append(sep); doc.append(-1); sep += 1   # unique sentinel -> never matches across trips
    id2cell = {v: k for k, v in vocab.items()}
    return np.array(seq, dtype=np.int64), np.array(doc, dtype=np.int64), id2cell, C


def mine_popular(seq, doc, sa, lcp, C, minsup, use_hll=False, hll_p=12):
    """All popular contiguous substrings (support>=minsup). Returns list of (id_tuple, support)."""
    n = len(seq)
    maxL = int(lcp.max())
    out = {}
    hll_err = []  # (exact, est)
    for ell in range(2, maxL + 1):
        i = 0
        while i < n:
            j = i
            while j + 1 < n and lcp[j + 1] >= ell:
                j += 1
            if j > i:
                start = sa[i]
                if start + ell <= n:
                    sub = seq[start:start + ell]
                    if sub.max() < C:                       # no sentinel -> single-trip substring
                        docs = doc[sa[i:j + 1]]
                        docs = docs[docs >= 0]
                        exact = len(np.unique(docs))
                        if use_hll:
                            h = HLL(hll_p)
                            for d in np.unique(docs):
                                h.add_int(int(d))
                            est = h.count()
                            hll_err.append((exact, est))
                            supp = est
                        else:
                            supp = exact
                        if supp >= minsup:
                            out[tuple(int(x) for x in sub)] = exact   # store exact support
            i = j + 1
    return out, hll_err


def maximal_filter(popular_keys):
    """Same maximality as method A: drop keys that are a prefix or suffix of a longer popular key."""
    extendable = set()
    for k in popular_keys:
        if len(k) >= 3:
            extendable.add(k[:-1]); extendable.add(k[1:])
        elif len(k) == 2:
            extendable.add((k[0],)); extendable.add((k[1],))
    return [k for k in popular_keys if k not in extendable]


def route_km_ids(id_tuple, id2cell):
    pts = [h3.cell_to_latlng(id2cell[i]) for i in id_tuple]
    t = 0.0
    for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
        dlat, dlon = radians(la2 - la1), radians(lo2 - lo1)
        a = sin(dlat / 2) ** 2 + cos(radians(la1)) * cos(radians(la2)) * sin(dlon / 2) ** 2
        t += 2 * 6371 * asin(sqrt(a))
    return float(t)
