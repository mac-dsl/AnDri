import numpy as np
import stumpy as st
import pandas as pd


class DAMP():
    """
    Implementation of the DAMP algorithm proposed in this paper https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf.
    Inspiration from https://github.com/HPI-Information-Systems/TimeEval-algorithms/blob/main/damp/damp/damp.py
    """

    """
    ----------
    m : int,
        target subsequence length.
    sp_index : int
        Need to be strictly greater than m. 
    x_lag : int
        default value None and set to 2**int(np.ceil(np.log2( 8*self.m )))
    ----------
    """

    def __init__(self,m,sp_index,x_lag = None, normalize='z-norm'):
        
        self.m = m
        self.sp_index = sp_index
        self.normalize = normalize
        if x_lag is not None:
            self.x_lag = x_lag
        else:
            self.x_lag = 2**int(np.ceil(np.log2( 8*self.m )))
        self._BFS = 0

    def fit(self, X, y=None):
        
        self._pv = np.ones(len(X) - self.m + 1, dtype=int)
        aMP = np.zeros_like(self._pv, dtype=float)

        for i in range(self.sp_index, len(X) - self.m + 1):
            if not self._pv[i]:
                aMP[i] = aMP[i-1]
            else:
                aMP[i] = self._BackwardProcessing(X, i)
                self._ForwardProcessing(X, i)

        self.decision_scores_ = aMP
        return self


    def _BackwardProcessing(self, X, i):
        aMP_i = np.inf
        prefix = 2** int(np.ceil(np.log2(self.m)))
        max_lag = min(self.x_lag or i, i)
        reference_ts = X[i-max_lag:i]
        first = True
        expansion_num = 0

        while aMP_i >= self._BFS:
            if prefix >= max_lag:
                ## Add normalize option
                # aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts, normalize=True))
                if self.normalize == 'z-norm':
                    aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts, normalize=True))
                elif self.normalize == 'euclidean':
                    aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts, normalize=False))
                elif self.normalize == 'zero-mean':
                    reference_ts_n = reference_ts - np.mean(reference_ts)
                    x_test_n = X[i:i+self.m] - np.mean(X[i:i+self.m])
                    aMP_i = min(st.core.mass(x_test_n, reference_ts_n, normalize=False))
                else:
                    aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts, normalize=True))
                if aMP_i > self._BFS:
                    self._BFS = aMP_i
                break
            else:
                if first:
                    first = False
                    # aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts[-prefix:], normalize=True))
                    if self.normalize == 'z-norm':
                        aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts[-prefix:], normalize=True))
                    elif self.normalize == 'euclidean':
                        aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts[-prefix:], normalize=False))
                    elif self.normalize == 'zero-mean':
                        reference_ts_n = reference_ts - np.mean(reference_ts)
                        x_test_n = X[i:i+self.m] - np.mean(X[i:i+self.m])
                        aMP_i = min(st.core.mass(x_test_n, reference_ts_n[-prefix:], normalize=False))
                    else:
                        aMP_i = min(st.core.mass(X[i:i+self.m], reference_ts[-prefix:], normalize=True))
                else:
                    start = i-max_lag+(expansion_num * self.m)
                    end = int(i-(max_lag/2)+(expansion_num * self.m))
                    # aMP_i = min(st.core.mass(X[i:i+self.m], X[start:end], normalize=True))
                    if self.normalize == 'z-norm':
                        aMP_i = min(st.core.mass(X[i:i+self.m], X[start:end], normalize=True))
                    elif self.normalize == 'euclidean':
                        aMP_i = min(st.core.mass(X[i:i+self.m], X[start:end], normalize=False))
                    elif self.normalize == 'zero-mean':
                        reference_ts_n = X[start:end] - np.mean(X[start:end])
                        x_test_n = X[i:i+self.m] - np.mean(X[i:i+self.m])
                        aMP_i = min(st.core.mass(x_test_n, reference_ts_n, normalize=False))
                    else:
                        aMP_i = min(st.core.mass(X[i:i+self.m], X[start:end], normalize=True))

                if aMP_i < self._BFS:
                    break
                else:
                    prefix = 2*prefix
                    expansion_num *= 1

        return aMP_i

    def _ForwardProcessing(self, X, i):
        start = i + self.m
        lookahead = 2** int(np.ceil(np.log2(self.m)))
        end = start + lookahead
        indices = []

        if end < len(X):
            if self.normalize == 'z-norm':
                d = st.core.mass(X[i:i+self.m], X[start:end], normalize=True)
            elif self.normalize == 'euclidean':
                d = st.core.mass(X[i:i+self.m], X[start:end], normalize=False)
            elif self.normalize == 'zero-mean':
                reference_ts_n = X[start:end] - np.mean(X[start:end])
                x_test_n = X[i:i+self.m] - np.mean(X[i:i+self.m])
                d = st.core.mass(x_test_n, reference_ts_n, normalize=False)
            indices = np.argwhere(d < self._BFS)
            indices += start

        self._pv[indices] = 0
