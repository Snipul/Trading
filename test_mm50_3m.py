# -*- coding: utf-8 -*-
"""Tests du bot MM50 trimestrielle : python -m pytest test_mm50_3m.py -v"""

import pandas as pd
import pytest

import mm50_3m as mm
from mm50_3m import APPROCHE, CONTACT, LOIN


def cadre_plat(debut="2012-01-02", fin="2026-09-25", prix=100.0):
    """Serie daily a prix constant, OHLC identiques."""
    idx = pd.bdate_range(debut, fin)
    return pd.DataFrame(
        {"Open": prix, "High": prix, "Low": prix, "Close": prix, "Volume": 1000.0},
        index=idx,
    )


def mesure(dist_close, dist_bas=None):
    if dist_bas is None:
        dist_bas = dist_close
    return {"dist_close": dist_close, "dist_bas": dist_bas}


# --- Calcul ----------------------------------------------------------------

def test_trimestre_en_cours_inclus():
    cadre = cadre_plat()
    trim = mm.to_quarterly(cadre)
    assert trim.index[-1] == pd.Timestamp("2026-09-30")  # trimestre en cours conserve
    assert trim.index[0] == pd.Timestamp("2012-03-31")


def test_mm50_integre_la_cloture_du_jour():
    cadre = cadre_plat()
    cadre.iloc[-1, cadre.columns.get_loc("Close")] = 150.0
    m = mm.mesurer(cadre)
    assert m["mm50"] == pytest.approx((49 * 100 + 150) / 50)


def test_moins_de_50_trimestres_ignore():
    assert mm.mesurer(cadre_plat(debut="2015-01-02")) is None


def test_distances():
    cadre = cadre_plat()
    cadre.iloc[-1, cadre.columns.get_loc("Low")] = 99.0
    m = mm.mesurer(cadre)
    assert m["dist_close"] == pytest.approx(0.0)
    assert m["dist_bas"] == pytest.approx(-1.0)


# --- Machine a etats -------------------------------------------------------

def test_titre_inconnu_sans_alerte():
    assert mm.evaluer(None, mesure(20)) == (LOIN, None)
    assert mm.evaluer(None, mesure(4)) == (APPROCHE, None)
    assert mm.evaluer(None, mesure(-3)) == (CONTACT, None)


def test_approche_une_seule_fois():
    assert mm.evaluer(LOIN, mesure(5.5)) == (APPROCHE, APPROCHE)
    assert mm.evaluer(APPROCHE, mesure(3)) == (APPROCHE, None)
    assert mm.evaluer(APPROCHE, mesure(8)) == (APPROCHE, None)  # entre 6 et 10 : pas de rearmement


def test_contact_rebond_dans_la_seance():
    # le plus bas touche la zone, la cloture repart bien au-dessus
    assert mm.evaluer(APPROCHE, mesure(3, dist_bas=0.3)) == (CONTACT, CONTACT)


def test_contact_cassure():
    assert mm.evaluer(APPROCHE, mesure(-2, dist_bas=-3)) == (CONTACT, CONTACT)


def test_contact_direct_sans_approche():
    # gap baissier depuis loin : l'alerte contact part quand meme
    assert mm.evaluer(LOIN, mesure(-4, dist_bas=-5)) == (CONTACT, CONTACT)


def test_contact_meme_si_cloture_au_dela_du_rearmement():
    assert mm.evaluer(APPROCHE, mesure(11, dist_bas=0.2)) == (CONTACT, CONTACT)


def test_pas_de_retrigger_contact():
    assert mm.evaluer(CONTACT, mesure(-5)) == (CONTACT, None)
    assert mm.evaluer(CONTACT, mesure(4)) == (CONTACT, None)
    assert mm.evaluer(CONTACT, mesure(0.2)) == (CONTACT, None)


def test_rearmement_au_dela_de_10():
    assert mm.evaluer(CONTACT, mesure(10.5)) == (LOIN, None)
    assert mm.evaluer(APPROCHE, mesure(12)) == (LOIN, None)
    assert mm.evaluer(LOIN, mesure(5)) == (APPROCHE, APPROCHE)


def test_libelle_contact():
    assert "rebond" in mm.libelle_contact(mesure(2))
    assert "dans la zone" in mm.libelle_contact(mesure(-0.3))
    assert "cassure" in mm.libelle_contact(mesure(-1))
