import numpy as np
import pandas as pd

def argmax_cache_linear(cache, A, V):
    y_st = -1
    delta_st = -1
    for y in V:
        if y in A:
            continue
        delta_y = cache[y]
        if delta_st < delta_y:
            delta_st = delta_y
            y_st = y
    assert y_st >= 0

    return y_st

def make_slice(cov_vv, y, A):
    cov_yA = np.zeros(shape=[len(y), len(A)])
    for i in range(cov_yA.shape[0]):
        for j in range(cov_yA.shape[1]):
            cov_yA[i,j] = cov_vv[y[i], A[j]]
    return cov_yA

def call_pinv(a):
    assert a.shape[0] == a.shape[1]
    if a.shape[0] == 1:
        return 1 / a
    else:
        r = np.linalg.pinv(a)
        return r

def nominator(y, A, cov_vv):
    A_ = list(A.copy())
    sigm_yy = make_slice(cov_vv, [y], [y])

    if 0 == len(A):
        retv = sigm_yy
    else:
        cov_yA = make_slice(cov_vv, [y], A_)
        cov_AA = make_slice(cov_vv, A_, A_)
        cov_Ay = make_slice(cov_vv, A_, [y])
        inv_cov_AA = call_pinv(cov_AA)
        dot_yA_iAA = np.dot(cov_yA, inv_cov_AA)
        dot_yAiAA_Ay = np.dot(dot_yA_iAA, cov_Ay)
        retv = sigm_yy - dot_yAiAA_Ay
    return retv

def denominator(y, A_hat, cov_vv):
    A_hat_ = list(A_hat.copy())
    if y in A_hat_:
        A_hat_.remove(int(y))
    return nominator(y, A_hat_, cov_vv)

def mi_placement_algo(cov_vv, k, init, pool_cnt):

    A = init.tolist() # selected_indexes
    V = np.linspace(0, cov_vv.shape[0]-1, cov_vv.shape[0], dtype=int)
    A_bar = [] # complementer set to A.
    delta_cached = []
    delta_cached_is_uptodate = []
    INF = 1e1000

    for i in V:
        if i not in A:
            A_bar.append(i)
        delta_cached.append(INF)
        delta_cached_is_uptodate.append(0)

    while len(A) < k+len(init):
        for i in range(len(delta_cached_is_uptodate)):
            delta_cached_is_uptodate[i] = False

        while True:
            y_st = argmax_cache_linear(delta_cached, A, V[:pool_cnt])
            if delta_cached_is_uptodate[y_st]:
                print('y*=', y_st)
                break

            delta_y = 0
            nom = nominator(y_st, A, cov_vv)
            denom = denominator(y_st, A_bar, cov_vv)

            if not ( np.abs(denom) < 1e-8 or np.abs(nom) < 1e-8 ):
                delta_y = nom / denom

            delta_cached[y_st] = delta_y
            delta_cached_is_uptodate[y_st] = True

        A.append(y_st)
        A_bar.remove(y_st)

    return A


def mi_placement(G, k):
    # First populate for S (candidate locations)
    cols = ['lat', 'long', 'time', 'pm']
    w = 'time'
    d = pd.DataFrame(G.pool)
    d.columns = cols
    d = d.groupby(['lat', 'long', w]).pm.mean().reset_index()
    D = d.pivot_table(index=['lat', 'long'], columns=[w], values='pm').reset_index()

    df = None
    for loc in G.loc_pool:
        F = np.logical_and(D.lat == loc[0], D.long == loc[1])
        assert np.sum(F) == 1
        df = pd.concat((df,D[F]))

    # Now populate for U (other / validation locations)
    d = pd.DataFrame(G.valid)
    d.columns = cols
    D = d.pivot_table(index=['lat', 'long'], columns=[w], values='pm').reset_index()
    df = pd.concat((df, D))

    # Now get the covariance
    data = df.values[:, 2:]
    mdf = np.ma.MaskedArray(data, np.isnan(data))
    cov = np.ma.cov(mdf, rowvar=True).data

    init = np.array([i for i in range(len(G.loc_init))])

    ans = mi_placement_algo(cov, k, init, len(G.loc_pool))
    print('mi', *ans)

    return ans[len(G.loc_init):]
