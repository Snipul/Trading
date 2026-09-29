import numpy as np
import pandas as pd

import backtest_mm50_swing as bt


def cadre_synthetique(n=700, seed=0, vol=0.015):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, vol, n)
    close = 50 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2019-01-01", periods=n)
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 1e6}, index=idx)


def test_pas_de_lookahead_sur_les_touches():
    cadre = cadre_synthetique()
    complet = bt.detecter_touches(cadre)
    assert len(complet) > 5
    for _, ligne in complet.iloc[:10].iterrows():
        coupe = bt.detecter_touches(cadre.iloc[: int(ligne["i"]) + 1])
        d = coupe.iloc[-1]
        assert d["date"] == ligne["date"]
        assert d["n_touche"] == ligne["n_touche"]
        assert d["tendance"] == ligne["tendance"]
        assert d["confirme"] == ligne["confirme"]


def test_touches_consecutives_comptent_pour_une():
    t = bt.detecter_touches(cadre_synthetique())
    assert (t["i"].diff().dropna() >= 2).all()


def test_simuler_stop_cible_et_gap():
    o = np.array([100, 100, 100, 100.0]); h = np.array([101, 101, 111, 101.0])
    b = np.array([99, 99, 99, 99.0]); c = np.array([100, 100, 100, 100.0])
    ret, R, jours, motif = bt.simuler(o, h, b, c, 0, 100.0, 5.0, 2.0, 3)
    assert motif == "cible" and abs(ret - 10) < 1e-9 and abs(R - 2) < 1e-9 and jours == 2
    b2 = np.array([99, 99, 90, 99.0])
    assert bt.simuler(o, h, b2, c, 0, 100.0, 5.0, 2.0, 3)[3] == "stop"
    o3 = np.array([100, 100, 90, 100.0]); b3 = np.array([99, 99, 88, 99.0]); h3 = np.array([101, 101, 95, 101.0])
    ret, R, jours, motif = bt.simuler(o3, h3, b3, c, 0, 100.0, 5.0, 2.0, 3)
    assert motif == "stop_gap" and abs(ret + 10) < 1e-9


def test_vectorise_identique_au_scalaire():
    """La version vectorisee doit reproduire la version scalaire, y compris en fin de donnees."""
    cadre = cadre_synthetique(n=500, seed=3, vol=0.03)
    o, h, b, c = (cadre[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    rng = np.random.default_rng(1)
    i_ent = rng.integers(10, len(c) - 1, 400)
    prix = o[i_ent]
    stops = rng.uniform(1, 12, 400)
    for rr, hold in ((2.0, 40), (1.5, 10), (3.0, 60)):
        ret, R, jours, ok = bt.simuler_vec(o, h, b, c, i_ent, prix, stops, rr, hold)
        for k in range(400):
            attendu = bt.simuler(o, h, b, c, int(i_ent[k]), float(prix[k]), float(stops[k]), rr, hold)
            if attendu is None:
                assert not ok[k]
            else:
                assert ok[k]
                assert abs(ret[k] - attendu[0]) < 1e-9
                assert abs(R[k] - attendu[1]) < 1e-9
                assert jours[k] == attendu[2]


def test_marche_utilise_la_veille():
    """L'etat du marche au jour t ne doit dependre que de donnees <= t-1."""
    indice = cadre_synthetique(n=600, seed=5)
    cible = indice.index
    complet = bt.serie_marche(indice, cible)
    t = 450
    coupe = bt.serie_marche(indice.iloc[:t], cible[:t])
    assert np.array_equal(complet[:t], coupe, equal_nan=True)
    # et un choc sur la cloture du jour t ne change pas l'etat du jour t
    modif = indice.copy()
    modif.iloc[t, modif.columns.get_loc("Close")] *= 0.2
    assert bt.serie_marche(modif, cible)[t] == complet[t]


def test_stop_atr_positif_et_borne():
    prix = np.array([100.0, 100.0, 100.0])
    atr = np.array([2.0, 0.0001, 80.0])
    s = bt.stop_effectif(("ATR x2", "atr", 2.0), prix, atr)
    assert abs(s[0] - 4.0) < 1e-9
    assert s[1] == bt.STOP_MIN_PCT and s[2] == bt.STOP_MAX_PCT
    assert (bt.stop_effectif(("fixe 5%", "pct", 5.0), prix, atr) == 5.0).all()


def test_atr_sans_lookahead():
    cadre = cadre_synthetique(n=300)
    a = bt.calcul_atr(cadre)
    a_coupe = bt.calcul_atr(cadre.iloc[:200])
    assert np.allclose(a[:200], a_coupe, equal_nan=True)


def test_motif_exclusion():
    cadre = cadre_synthetique(n=300)
    assert bt.motif_exclusion(cadre)[0] is None
    saut = cadre.copy()
    saut.iloc[150:, :4] *= 4.0
    assert bt.motif_exclusion(saut)[0] == "saut de prix"
    trou = cadre.drop(cadre.index[100:150])
    assert bt.motif_exclusion(trou)[0] == "trou de cotation"


def test_stats_agreg_coherent_avec_stats_trades():
    df = pd.DataFrame({"ret": [5.0, -2.0, 3.0, -1.0], "R": [1.0, -0.4, 0.6, -0.2], "jours": [3, 2, 5, 1]})
    st = bt.stats_trades(df)
    g = pd.DataFrame([{"n": 4, "s_ret": df.ret.sum(), "s_R": df.R.sum(), "s_jours": df.jours.sum(),
                       "wins": 2, "gains": 8.0, "pertes": 3.0}])
    sa = bt.stats_agreg(g)
    for k in ("n", "gagnants %", "ret moy %", "R moy", "PF", "jours moy"):
        assert abs(st[k] - sa[k]) < 1e-9
