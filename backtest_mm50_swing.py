#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_mm50_swing.py (v2) — Backtest du setup swing « rebond sur la MM50 daily »

Question testee
---------------
Prix au-dessus d'une MM200 daily ASCENDANTE, le cours revient par le haut sur
la MM50 daily : le 1er (ou 2e) touche rebondit-il mieux que les suivants ?
Et mieux qu'un jour quelconque de la meme tendance, A SORTIE IDENTIQUE ?

Definitions (donnees connues AVANT la seance du touche)
-------------------------------------------------------
Tendance   : cloture(t-1) > MM200(t-1) ET MM200(t-1) > MM200(t-1-20).
Marche     : indice de reference > sa propre MM200 a la cloture t-1
             (US : GSPC.INDX ; Euronext : STOXX50E.INDX).
Zone       : MM50(t-1) * (1 + tol). Touche = plus bas(t) <= zone, la veille
             etant entierement au-dessus. Des touches consecutives = 1 seul
             touche. Le compteur repart a 0 quand la cloture passe au-dessus
             de MM50 * (1 + reset).

Deux facons d'entrer
--------------------
limit     : ordre limite sur la zone. Prix = min(ouverture, zone), jour du touche.
confirme  : touche + cloture > MM50 + bougie verte. Entree a l'ouverture suivante.

Sorties (par trade)
-------------------
Stop fixe en % (--stops) OU stop en multiple d'ATR(14) (--atr, connu la veille
de l'entree). Objectif = +rr x distance du stop, sortie temps a max_hold
seances. Stop et objectif le meme jour : stop retenu. Gap d'ouverture au-dela :
sortie a l'ouverture. Aucun frais ni slippage.

Reference : TOUS les jours en tendance, entree a l'ouverture suivante, avec
exactement les memes stops / objectif / sortie temps. C'est la comparaison
qui dit si le touche apporte un avantage propre.

Limites
-------
* Biais du survivant : univers d'aujourd'hui, titres disparus absents.
* Exclusions de donnees suspectes : titres ecartes en bloc, detail dans le
  rapport (section Exclusions) et dans excluded.csv.
* Trades d'un meme jour tres correles : ne pas lire n comme independants.

Utilisation
-----------
    python backtest_mm50_swing.py                          # US + Euronext, 10 ans
    python backtest_mm50_swing.py --echantillon 100 --graine 7
    python backtest_mm50_swing.py --stops 3 5 --atr 1.5 2 3
"""

import argparse
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (10, 20, 40)
BUCKETS = ["1er", "2e", "3e", "4e+"]
INDICE_US = "GSPC.INDX"
INDICE_EURO = "STOXX50E.INDX"
GAP_PRIX_MAX = 80.0
GAP_JOURS_MAX = 60
STOP_MIN_PCT, STOP_MAX_PCT = 0.5, 30.0


# ---------------------------------------------------------------------------
# Indicateurs
# ---------------------------------------------------------------------------

def calcul_atr(cadre, periode=14):
    """ATR simple. atr[t] utilise les seances <= t."""
    h, b, c = cadre["High"], cadre["Low"], cadre["Close"]
    cp = c.shift(1)
    tr = pd.concat([h - b, (h - cp).abs(), (b - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(periode).mean().to_numpy()


def serie_marche(indice, index_cible):
    """
    Booleen 'indice > MM200' connu a la cloture de la VEILLE, aligne sur les
    dates du titre (dernier etat connu). NaN si pas assez d'historique.
    """
    ma = indice["Close"].rolling(200).mean()
    etat = (indice["Close"] > ma).where(ma.notna()).astype(float).shift(1)
    return etat.reindex(index_cible, method="ffill").to_numpy()


# ---------------------------------------------------------------------------
# Signaux
# ---------------------------------------------------------------------------

def detecter_touches(cadre, tol=0.005, reset=0.08, pente_jours=20):
    """
    DataFrame des touches : date, i, n_touche, tendance, confirme, zone.
    Aucune information posterieure a la seance du touche n'est utilisee.
    """
    c = cadre["Close"].to_numpy(float)
    o = cadre["Open"].to_numpy(float)
    b = cadre["Low"].to_numpy(float)
    ma50 = cadre["Close"].rolling(50).mean().to_numpy()
    ma200 = cadre["Close"].rolling(200).mean().to_numpy()

    lignes = []
    n = 0
    touche_veille = False
    depart = 200 + pente_jours + 1
    for t in range(depart, len(c)):
        zone = ma50[t - 1] * (1 + tol)
        zone_veille = ma50[t - 2] * (1 + tol)
        est_touche = b[t] <= zone
        venait_du_haut = b[t - 1] > zone_veille

        if c[t] > ma50[t] * (1 + reset):
            n = 0
            touche_veille = False
            continue

        if est_touche and not touche_veille and venait_du_haut:
            n += 1
            tendance = (c[t - 1] > ma200[t - 1]) and (ma200[t - 1] > ma200[t - 1 - pente_jours])
            confirme = (c[t] > ma50[t]) and (c[t] > o[t])
            lignes.append((cadre.index[t], t, n, bool(tendance), bool(confirme), zone))
        touche_veille = est_touche

    return pd.DataFrame(lignes, columns=["date", "i", "n_touche", "tendance", "confirme", "zone"])


def jours_en_tendance(cadre, pente_jours=20):
    """Indices t (entiers) des seances ou la tendance est valide (donnees <= t-1)."""
    c = cadre["Close"].to_numpy(float)
    ma200 = cadre["Close"].rolling(200).mean().to_numpy()
    t = np.arange(200 + pente_jours + 1, len(c) - 1)
    ok = (c[t - 1] > ma200[t - 1]) & (ma200[t - 1] > ma200[t - 1 - pente_jours])
    return t[ok]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simuler(o, h, b, c, i_entree, prix, stop_pct, rr, max_hold):
    """
    Version scalaire (reference de test). Long entre a `prix` a la seance
    i_entree ; stop et objectif testes des cette seance.
    Retourne (rendement %, R, jours, motif) ou None si donnees insuffisantes.
    """
    stop = prix * (1 - stop_pct / 100)
    cible = prix * (1 + rr * stop_pct / 100)
    fin = min(i_entree + max_hold, len(c) - 1)
    for k in range(i_entree, fin + 1):
        if k > i_entree:
            if o[k] <= stop:
                return _res(o[k], prix, stop_pct, k - i_entree, "stop_gap")
            if o[k] >= cible:
                return _res(o[k], prix, stop_pct, k - i_entree, "cible_gap")
        if b[k] <= stop:
            return _res(stop, prix, stop_pct, k - i_entree, "stop")
        if h[k] >= cible:
            return _res(cible, prix, stop_pct, k - i_entree, "cible")
    if fin - i_entree < max_hold:
        return None
    return _res(c[fin], prix, stop_pct, fin - i_entree, "temps")


def _res(sortie, prix, stop_pct, jours, motif):
    ret = (sortie / prix - 1) * 100
    return ret, ret / stop_pct, jours, motif


def simuler_vec(o, h, b, c, i_entree, prix, stop_pct, rr, max_hold):
    """
    Version vectorisee, memes regles que `simuler`. i_entree, prix, stop_pct :
    tableaux de meme longueur. Retourne (ret, R, jours, valide) ; ret/R/jours
    ne sont significatifs que ou valide est True.
    """
    n = len(c)
    m = len(i_entree)
    stop = prix * (1 - stop_pct / 100)
    cible = prix * (1 + rr * stop_pct / 100)
    ret = np.full(m, np.nan)
    jours = np.zeros(m, dtype=int)
    resolu = np.zeros(m, dtype=bool)

    for k in range(max_hold + 1):
        idx = i_entree + k
        dispo = idx < n
        j = np.where(dispo, idx, 0)
        actif = ~resolu & dispo

        if k > 0:
            for cond, niveau in ((actif & (o[j] <= stop), o[j]), (actif & (o[j] >= cible), o[j])):
                ret[cond] = (niveau[cond] / prix[cond] - 1) * 100
                jours[cond] = k
                resolu |= cond
                actif = actif & ~cond

        cond = actif & (b[j] <= stop)
        ret[cond] = (stop[cond] / prix[cond] - 1) * 100
        jours[cond] = k
        resolu |= cond
        actif = actif & ~cond

        cond = actif & (h[j] >= cible)
        ret[cond] = (cible[cond] / prix[cond] - 1) * 100
        jours[cond] = k
        resolu |= cond
        actif = actif & ~cond

        if k == max_hold:
            ret[actif] = (c[j][actif] / prix[actif] - 1) * 100
            jours[actif] = k
            resolu |= actif

    return ret, ret / stop_pct, jours, resolu


def configs_stop(stops, atrs):
    """Liste de (libelle, type, valeur)."""
    return [(f"fixe {s:g}%", "pct", s) for s in stops] + [(f"ATR x{k:g}", "atr", k) for k in atrs]


def stop_effectif(config, prix, atr_veille):
    """Distance du stop en % du prix d'entree (tableau)."""
    _, type_, valeur = config
    if type_ == "pct":
        return np.full(len(prix), float(valeur))
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = valeur * atr_veille / prix * 100
    return np.clip(pct, STOP_MIN_PCT, STOP_MAX_PCT)


def rendements_termes(c, i_ref, prix):
    """Rendements (cloture) a HORIZONS seances depuis l'entree, NaN hors donnees."""
    out = {}
    n = len(c)
    for hz in HORIZONS:
        j = i_ref + hz
        ok = j < n
        r = np.full(len(i_ref), np.nan)
        r[ok] = (c[j[ok]] / prix[ok] - 1) * 100
        out[f"r{hz}"] = r
    return out


# ---------------------------------------------------------------------------
# Backtest d'un ticker
# ---------------------------------------------------------------------------

def trades_ticker(ticker, place, cadre, marche, debut, configs, rr, max_hold, tol, reset):
    """Retourne (trades par config, termes par touche/variante)."""
    o = cadre["Open"].to_numpy(float)
    h = cadre["High"].to_numpy(float)
    b = cadre["Low"].to_numpy(float)
    c = cadre["Close"].to_numpy(float)
    atr = calcul_atr(cadre)
    dates = cadre.index

    touches = detecter_touches(cadre, tol=tol, reset=reset)
    if touches.empty:
        return [], []
    touches = touches[touches["date"] >= debut]
    if touches.empty:
        return [], []

    i = touches["i"].to_numpy()
    zone = touches["zone"].to_numpy()
    lots = [("limit", i, np.minimum(o[i], zone), np.ones(len(i), dtype=bool))]
    suivant = i + 1
    ok_conf = touches["confirme"].to_numpy() & (suivant < len(c))
    lots.append(("confirme", suivant[ok_conf], o[suivant[ok_conf]], ok_conf))

    trades, termes = [], []
    for variante, i_ent, prix, masque in lots:
        sel = touches[masque]
        if len(sel) == 0:
            continue
        valide = np.isfinite(prix) & (prix > 0) & np.isfinite(atr[i_ent - 1])
        if not valide.any():
            continue
        i_ent, prix, sel = i_ent[valide], prix[valide], sel[valide]
        atr_v = atr[i_ent - 1]
        marche_t = marche[sel["i"].to_numpy()] if marche is not None else np.full(len(sel), np.nan)

        base = pd.DataFrame({
            "ticker": ticker, "place": place, "date": sel["date"].to_numpy(),
            "n_touche": sel["n_touche"].to_numpy(), "tendance": sel["tendance"].to_numpy(),
            "marche": marche_t, "variante": variante,
        })
        termes.append(pd.concat([base, pd.DataFrame(rendements_termes(c, i_ent, prix))], axis=1))

        for config in configs:
            s = stop_effectif(config, prix, atr_v)
            bon = np.isfinite(s)
            if not bon.any():
                continue
            ret, R, jours, ok = simuler_vec(o, h, b, c, i_ent[bon], prix[bon], s[bon], rr, max_hold)
            t = base[bon][ok].copy()
            t["config"] = config[0]
            t["stop_eff"] = s[bon][ok]
            t["ret"], t["R"], t["jours"] = ret[ok], R[ok], jours[ok]
            trades.append(t)
    return trades, termes


def reference_ticker(place, cadre, marche, debut, configs, rr, max_hold):
    """
    Reference : tous les jours en tendance, entree a l'ouverture suivante,
    memes sorties. Retourne (agregats par annee/regime/config, termes journaliers).
    """
    o = cadre["Open"].to_numpy(float)
    h = cadre["High"].to_numpy(float)
    b = cadre["Low"].to_numpy(float)
    c = cadre["Close"].to_numpy(float)
    atr = calcul_atr(cadre)
    dates = cadre.index

    t = jours_en_tendance(cadre)
    t = t[dates[t] >= debut]
    if len(t) == 0:
        return [], None
    i_ent = t + 1
    prix = o[i_ent]
    ok = np.isfinite(prix) & (prix > 0) & np.isfinite(atr[t])
    t, i_ent, prix = t[ok], i_ent[ok], prix[ok]
    if len(t) == 0:
        return [], None
    atr_v = atr[t]
    marche_t = marche[t] if marche is not None else np.full(len(t), np.nan)
    annee = dates[t].year.to_numpy()

    termes = pd.DataFrame({"place": place, "marche": marche_t, **rendements_termes(c, i_ent, prix)})

    agregats = []
    for config in configs:
        s = stop_effectif(config, prix, atr_v)
        bon = np.isfinite(s)
        if not bon.any():
            continue
        ret, R, jours, valide = simuler_vec(o, h, b, c, i_ent[bon], prix[bon], s[bon], rr, max_hold)
        d = pd.DataFrame({"annee": annee[bon][valide], "marche": marche_t[bon][valide],
                          "ret": ret[valide], "R": R[valide], "jours": jours[valide]})
        d["gain"] = d["ret"].clip(lower=0)
        d["perte"] = (-d["ret"]).clip(lower=0)
        d["win"] = (d["ret"] > 0).astype(int)
        g = d.groupby(["annee", "marche"], dropna=False).agg(
            n=("ret", "size"), s_ret=("ret", "sum"), s_R=("R", "sum"), s_jours=("jours", "sum"),
            wins=("win", "sum"), gains=("gain", "sum"), pertes=("perte", "sum")).reset_index()
        g["place"], g["config"] = place, config[0]
        agregats.append(g)
    return agregats, termes


# ---------------------------------------------------------------------------
# Donnees : exclusions
# ---------------------------------------------------------------------------

def motif_exclusion(cadre, gap_prix_max=GAP_PRIX_MAX, gap_jours_max=GAP_JOURS_MAX):
    """
    Meme logique que stromboli._donnee_suspecte, mais avec le motif.
    Retourne (motif ou None, plus grand saut %, plus grand trou en jours).
    """
    closes = cadre["Close"].to_numpy(float)
    saut = 0.0
    trou = 0.0
    if len(closes) >= 2:
        v = np.abs(np.diff(closes) / closes[:-1] * 100)
        saut = float(np.nanmax(v)) if len(v) else 0.0
        e = cadre.index.to_series().diff().dt.days.to_numpy()[1:]
        trou = float(np.nanmax(e)) if len(e) else 0.0
    if saut > gap_prix_max and trou > gap_jours_max:
        return "saut de prix + trou de cotation", saut, trou
    if saut > gap_prix_max:
        return "saut de prix", saut, trou
    if trou > gap_jours_max:
        return "trou de cotation", saut, trou
    return None, saut, trou


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def bucket(n):
    return BUCKETS[min(n, 4) - 1]


def stats_trades(df):
    if df.empty:
        return {"n": 0}
    gains = df.loc[df["ret"] > 0, "ret"].sum()
    pertes = -df.loc[df["ret"] < 0, "ret"].sum()
    return {
        "n": len(df),
        "gagnants %": round((df["ret"] > 0).mean() * 100, 1),
        "ret moy %": round(df["ret"].mean(), 2),
        "R moy": round(df["R"].mean(), 3),
        "PF": round(gains / pertes, 2) if pertes > 0 else np.inf,
        "jours moy": round(df["jours"].mean(), 1),
    }


def stats_agreg(g):
    """Memes stats a partir des sommes de la reference."""
    n = g["n"].sum()
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "gagnants %": round(g["wins"].sum() / n * 100, 1),
        "ret moy %": round(g["s_ret"].sum() / n, 2),
        "R moy": round(g["s_R"].sum() / n, 3),
        "PF": round(g["gains"].sum() / g["pertes"].sum(), 2) if g["pertes"].sum() > 0 else np.inf,
        "jours moy": round(g["s_jours"].sum() / n, 1),
    }


def tableau(df, par):
    lignes = []
    for cle, groupe in df.groupby(par, sort=False, dropna=False):
        cle = cle if isinstance(cle, tuple) else (cle,)
        lignes.append({**dict(zip(par, cle)), **stats_trades(groupe)})
    return pd.DataFrame(lignes)


def md(df):
    return df.to_markdown(index=False) if len(df) else "_aucune donnee_"


def _libelle_marche(x):
    return {1.0: "marche haussier", 0.0: "marche baissier"}.get(x, "inconnu")


def rapport(trades, termes, ref_agr, ref_termes, args, configs, exclusions, info):
    S = []
    ajouter = S.append
    noms = [c[0] for c in configs]
    detail = args.detail if args.detail in noms else noms[len(noms) // 2]
    marche_dispo = trades["marche"].notna().any()

    ajouter("# Backtest MM50 daily — rebond en tendance (MM200 ascendante) — v2\n")
    ajouter(f"- Fenetre : {args.annees} ans · {info}")
    ajouter(f"- Zone de touche : MM50 + {args.tol * 100:.1f} % · reset : cloture > MM50 + {args.reset * 100:.0f} %")
    ajouter(f"- Sorties testees : {', '.join(noms)} · objectif +{args.rr:g}R · sortie temps {args.max_hold} seances · sans frais")
    ajouter(f"- Config de detail (sections 4 a 6) : **{detail}**")
    ajouter(f"- Filtre de marche : {'actif (US : ' + INDICE_US + ', Euronext : ' + INDICE_EURO + ')' if marche_dispo else 'INDISPONIBLE (indice non telecharge)'}\n")

    tend = trades[trades["tendance"]].copy()
    tend["touche"] = tend["n_touche"].map(bucket)
    tend["groupe"] = np.where(tend["n_touche"] <= 2, "1er-2e", "3e+")
    tt = termes[termes["tendance"]].copy()
    tt["touche"] = tt["n_touche"].map(bucket)

    # 1 -----------------------------------------------------------------
    ajouter("## 1. Rendement brut a horizon fixe (sans stop), touches en tendance vs reference\n")
    for variante in ("limit", "confirme"):
        ajouter(f"### Entree `{variante}`")
        lignes = []
        for nom in BUCKETS:
            g = tt[(tt["variante"] == variante) & (tt["touche"] == nom)]
            for hz in HORIZONS:
                col = g[f"r{hz}"].dropna()
                if len(col):
                    lignes.append({"touche": nom, "horizon": f"{hz}j", "n": len(col),
                                   "ret moy %": round(col.mean(), 2), "mediane %": round(col.median(), 2),
                                   "% positifs": round((col > 0).mean() * 100, 1)})
        ajouter(md(pd.DataFrame(lignes)) + "\n")

    ajouter("### Reference (n'importe quel jour en tendance)")
    lignes = []
    for hz in HORIZONS:
        col = ref_termes[f"r{hz}"].dropna()
        lignes.append({"reference": "tous jours en tendance", "horizon": f"{hz}j", "n": len(col),
                       "ret moy %": round(col.mean(), 2), "mediane %": round(col.median(), 2),
                       "% positifs": round((col > 0).mean() * 100, 1)})
    ajouter(md(pd.DataFrame(lignes)) + "\n")

    # 2 -----------------------------------------------------------------
    ajouter("## 2. Avec stop / objectif : touches vs REFERENCE A SORTIE IDENTIQUE\n")
    ajouter("Excedent = R moyen du touche moins R moyen de la reference (meme config). "
            "Un excedent proche de 0 = le touche n'apporte rien de propre.\n")
    for variante in ("limit", "confirme"):
        ajouter(f"### Entree `{variante}`")
        lignes = []
        for nom in noms:
            r_ref = stats_agreg(ref_agr[ref_agr["config"] == nom]).get("R moy", np.nan)
            for groupe in ("1er-2e", "3e+"):
                g = tend[(tend["variante"] == variante) & (tend["config"] == nom) & (tend["groupe"] == groupe)]
                st = stats_trades(g)
                if st["n"]:
                    lignes.append({"config": nom, "groupe": groupe, **st,
                                   "R reference": r_ref, "excedent R": round(st["R moy"] - r_ref, 3)})
        ajouter(md(pd.DataFrame(lignes)) + "\n")

    lignes = []
    for nom in noms:
        lignes.append({"config": nom, **stats_agreg(ref_agr[ref_agr["config"] == nom])})
    ajouter("### Reference : stats par config")
    ajouter(md(pd.DataFrame(lignes)) + "\n")

    # 3 -----------------------------------------------------------------
    ajouter("## 3. Detail par n° de touche (entree `limit`)\n")
    sous = tend[tend["variante"] == "limit"]
    t = tableau(sous, ["config", "touche"])
    if len(t):
        t["ordre"] = t["touche"].map({n: k for k, n in enumerate(BUCKETS)})
        t["cfg"] = t["config"].map({n: k for k, n in enumerate(noms)})
        t = t.sort_values(["cfg", "ordre"]).drop(columns=["ordre", "cfg"])
    ajouter(md(t) + "\n")

    # 4 -----------------------------------------------------------------
    ajouter(f"## 4. Filtre de marche et filtre de tendance (config {detail}, entree `limit`)\n")
    d = trades[(trades["config"] == detail) & (trades["variante"] == "limit")].copy()
    d["groupe"] = np.where(d["n_touche"] <= 2, "1er-2e", "3e+")
    d["tendance_lbl"] = np.where(d["tendance"], "en tendance", "hors tendance")
    ajouter("### Effet du filtre de tendance (tous les touches)")
    ajouter(md(tableau(d, ["tendance_lbl"])) + "\n")
    if marche_dispo:
        dm = d[d["tendance"]].copy()
        dm["marche_lbl"] = dm["marche"].map(_libelle_marche)
        ajouter("### Touches en tendance, selon l'etat du marche (indice vs sa MM200)")
        lignes = []
        for lbl, g in dm.groupby("marche_lbl"):
            marche_val = {"marche haussier": 1.0, "marche baissier": 0.0}.get(lbl)
            rr_ = ref_agr[(ref_agr["config"] == detail)]
            rr_ = rr_[rr_["marche"] == marche_val] if marche_val is not None else rr_[rr_["marche"].isna()]
            r_ref = stats_agreg(rr_).get("R moy", np.nan)
            for groupe in ("1er-2e", "3e+"):
                st = stats_trades(g[g["groupe"] == groupe])
                if st["n"]:
                    lignes.append({"marche": lbl, "groupe": groupe, **st, "R reference": r_ref,
                                   "excedent R": round(st["R moy"] - r_ref, 3) if pd.notna(r_ref) else np.nan})
        ajouter(md(pd.DataFrame(lignes)) + "\n")

    # 5 -----------------------------------------------------------------
    ajouter(f"## 5. Robustesse par annee (config {detail}, entree `limit`, 1er-2e touche, en tendance)\n")
    an = d[d["tendance"] & (d["groupe"] == "1er-2e")].copy()
    an["annee"] = an["date"].dt.year
    lignes = []
    for annee, g in sorted(an.groupby("annee"), key=lambda x: x[0]):
        st = stats_trades(g)
        rg = ref_agr[(ref_agr["config"] == detail) & (ref_agr["annee"] == annee)]
        r_ref = stats_agreg(rg).get("R moy", np.nan)
        lignes.append({"annee": int(annee), **st, "R reference": r_ref,
                       "excedent R": round(st["R moy"] - r_ref, 3) if pd.notna(r_ref) else np.nan})
    ajouter(md(pd.DataFrame(lignes)) + "\n")

    # 6 -----------------------------------------------------------------
    ajouter(f"## 6. Par place (config {detail}, entree `limit`, 1er-2e touche, en tendance)\n")
    lignes = []
    for place, g in an.groupby("place", sort=False):
        st = stats_trades(g)
        r_ref = stats_agreg(ref_agr[(ref_agr["config"] == detail) & (ref_agr["place"] == place)]).get("R moy", np.nan)
        lignes.append({"place": place, **st, "R reference": r_ref,
                       "excedent R": round(st["R moy"] - r_ref, 3) if pd.notna(r_ref) else np.nan})
    ajouter(md(pd.DataFrame(lignes)) + "\n")

    # 7 -----------------------------------------------------------------
    ajouter("## 7. Exclusions de donnees par place\n")
    ajouter(md(exclusions) + "\n")
    ajouter("Detail des titres ecartes : `excluded.csv` (motif, plus grand saut, plus grand trou).\n")

    ajouter("## Lecture\n")
    ajouter("- Section 2 : c'est l'excedent de R vs reference qui compte, pas le R brut (les actions montent en moyenne).")
    ajouter("- Un excedent < 0,05 R est du bruit une fois frais et slippage payes.")
    ajouter("- Section 5 : un edge qui ne vit que sur quelques annees haussieres n'en est pas un.")
    ajouter("- Biais du survivant : resultats optimistes, surtout si beaucoup de titres sont exclus (section 7).")
    return "\n".join(S)


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Backtest MM50 daily en tendance (v2)")
    p.add_argument("--univers", default="tout", choices=["tout", "us", "euronext"])
    p.add_argument("--annees", type=int, default=10)
    p.add_argument("--stops", type=float, nargs="*", default=[3.0, 5.0, 8.0])
    p.add_argument("--atr", type=float, nargs="*", default=[1.5, 2.0, 3.0])
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--max-hold", type=int, default=40)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--reset", type=float, default=0.08)
    p.add_argument("--detail", default="ATR x2", help="config utilisee pour les sections de detail")
    p.add_argument("--echantillon", type=int, default=0,
                   help="tickers tires AU HASARD par place (0 = tous)")
    p.add_argument("--graine", type=int, default=42)
    p.add_argument("--sortie", default="backtest_out")
    args = p.parse_args()

    import stromboli as sb  # import tardif : le reste reste testable sans

    configs = configs_stop(args.stops, args.atr)
    debut = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.DateOffset(years=args.annees)
    periode = f"{args.annees + 1}y"
    sb.TWELVEDATA_API_KEY = ""

    univers = sb.construire_univers(args.univers)
    rng = random.Random(args.graine)

    # Indices de marche
    indices = sb.telecharger([INDICE_US, INDICE_EURO], periode)
    for nom in (INDICE_US, INDICE_EURO):
        if nom not in indices:
            print(f"ATTENTION : indice {nom} indisponible, filtre de marche desactive pour sa zone.")

    trades_l, termes_l, ref_l, ref_termes_l, excl_l, excl_stats = [], [], [], [], [], []
    n_ok = 0

    for place, tickers in univers.items():
        if place in ("Crypto", "Indices"):
            continue
        if args.echantillon and len(tickers) > args.echantillon:
            tickers = rng.sample(list(tickers), args.echantillon)
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})", flush=True)
        donnees = sb.telecharger(tickers, periode)
        cle_indice = INDICE_US if place == "US" else INDICE_EURO
        compt = {"place": place, "demandes": len(tickers), "sans donnees": len(tickers) - len(donnees),
                 "trop courts": 0, "saut de prix": 0, "trou de cotation": 0,
                 "saut + trou": 0, "exploitables": 0}

        for ticker, cadre in donnees.items():
            if len(cadre) < 260:
                compt["trop courts"] += 1
                excl_l.append({"ticker": ticker, "place": place, "motif": "trop court", "plus grand saut %": np.nan, "plus grand trou j": np.nan})
                continue
            motif, saut, trou = motif_exclusion(cadre)
            if sb._donnee_suspecte(cadre) or motif:
                motif = motif or "autre"
                cle = {"saut de prix": "saut de prix", "trou de cotation": "trou de cotation"}.get(motif, "saut + trou")
                compt[cle] += 1
                excl_l.append({"ticker": ticker, "place": place, "motif": motif,
                               "plus grand saut %": round(saut, 1), "plus grand trou j": trou})
                continue

            compt["exploitables"] += 1
            n_ok += 1
            marche = serie_marche(indices[cle_indice], cadre.index) if cle_indice in indices else None
            t, te = trades_ticker(ticker, place, cadre, marche, debut, configs, args.rr,
                                  args.max_hold, args.tol, args.reset)
            trades_l += t
            termes_l += te
            agr, rt = reference_ticker(place, cadre, marche, debut, configs, args.rr, args.max_hold)
            ref_l += agr
            if rt is not None:
                ref_termes_l.append(rt)
        excl_stats.append(compt)

    if not trades_l:
        print("Aucun trade : verifie EODHD_API_KEY et l'univers.")
        return 1

    trades = pd.concat(trades_l, ignore_index=True)
    termes = pd.concat(termes_l, ignore_index=True)
    ref_agr = pd.concat(ref_l, ignore_index=True)
    ref_termes = pd.concat(ref_termes_l, ignore_index=True)
    exclusions = pd.DataFrame(excl_stats)
    info = (f"{n_ok} titres exploitables sur {int(exclusions['demandes'].sum())} demandes "
            f"(voir section 7)")

    texte = rapport(trades, termes, ref_agr, ref_termes, args, configs, exclusions, info)
    print("\n" + texte)

    sortie = Path(args.sortie)
    sortie.mkdir(exist_ok=True)
    trades.to_csv(sortie / "trades.csv", index=False)
    pd.DataFrame(excl_l).to_csv(sortie / "excluded.csv", index=False)
    (sortie / "rapport.md").write_text(texte, encoding="utf-8")

    resume = os.getenv("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a", encoding="utf-8") as f:
            f.write(texte + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
