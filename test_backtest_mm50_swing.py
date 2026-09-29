import numpy as np
import pandas as pd

import backtest_mm50_swing as bt


def cadre_synthetique(n=700, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.015, n)
    close = 50 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2019-01-01", periods=n)
    op = close * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": 1e6}, index=idx)


def test_pas_de_lookahead_sur_les_touches():
    """Tronquer les donnees apres un touche ne doit pas changer ce touche."""
    cadre = cadre_synthetique()
    complet = bt.detecter_touches(cadre)
    assert len(complet) > 5
    for _, ligne in complet.iloc[:10].iterrows():
        coupe = bt.detecter_touches(cadre.iloc[: int(ligne["i"]) + 1])
        derniere = coupe.iloc[-1]
        assert derniere["date"] == ligne["date"]
        assert derniere["n_touche"] == ligne["n_touche"]
        assert derniere["tendance"] == ligne["tendance"]
        assert derniere["confirme"] == ligne["confirme"]


def test_touches_consecutives_comptent_pour_une():
    cadre = cadre_synthetique()
    t = bt.detecter_touches(cadre)
    ecarts = t["i"].diff().dropna()
    assert (ecarts >= 2).all()


def test_simuler_stop_cible_et_gap():
    o = np.array([100, 100, 100, 100.0]); h = np.array([101, 101, 111, 101.0])
    b = np.array([99, 99, 99, 99.0]); c = np.array([100, 100, 100, 100.0])
    # objectif +2R avec stop 5 % = +10 % : atteint au jour 2 (high 111)
    ret, R, jours, motif = bt.simuler(o, h, b, c, 0, 100.0, 5.0, 2.0, 3)
    assert motif == "cible" and abs(ret - 10) < 1e-9 and abs(R - 2) < 1e-9 and jours == 2
    # stop et cible le meme jour : stop retenu
    b2 = np.array([99, 99, 90, 99.0])
    assert bt.simuler(o, h, b2, c, 0, 100.0, 5.0, 2.0, 3)[3] == "stop"
    # gap sous le stop : sortie a l'ouverture
    o3 = np.array([100, 100, 90, 100.0]); b3 = np.array([99, 99, 88, 99.0]); h3 = np.array([101, 101, 95, 101.0])
    ret, R, jours, motif = bt.simuler(o3, h3, b3, c, 0, 100.0, 5.0, 2.0, 3)
    assert motif == "stop_gap" and abs(ret + 10) < 1e-9
