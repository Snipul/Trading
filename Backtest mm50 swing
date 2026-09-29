#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_mm50_swing.py — Backtest du setup swing « rebond sur la MM50 daily »

Question testee
---------------
Prix au-dessus d'une MM200 daily ASCENDANTE, le cours revient par le haut sur
la MM50 daily : le 1er (ou 2e) touche rebondit-il mieux que les suivants ?
Et mieux qu'un jour quelconque de la meme tendance (reference) ?

Definitions (toutes calculees avec des donnees connues AVANT la seance du touche)
-----------------------------------------------------------------------------
Tendance   : cloture(t-1) > MM200(t-1) ET MM200(t-1) > MM200(t-1-20).
Zone       : MM50(t-1) * (1 + tol). Touche = plus bas(t) <= zone, la veille
             etant entierement au-dessus (plus bas(t-1) > zone(t-1)).
N° de touche : compteur d'episodes. Des touches consecutives = 1 seul touche.
             Le compteur repart a 0 quand la cloture passe au-dessus de
             MM50 * (1 + reset) (vrai decollage). Compte meme hors tendance.

Deux facons d'entrer
--------------------
limit     : ordre limite pose sur la zone. Prix = min(ouverture, zone).
            Entree le jour du touche (le stop est teste des ce jour).
confirme  : touche + cloture > MM50 + bougie verte (cloture > ouverture).
            Entree a l'ouverture du lendemain.

Sortie (par trade)
------------------
Stop a -stop % sous l'entree, objectif a +rr x stop %, sortie au bout de
max_hold seances sinon. Stop et objectif touches le meme jour : le stop est
retenu (hypothese prudente). Gap d'ouverture au-dela du stop / de l'objectif :
sortie a l'ouverture. Aucun frais ni slippage.

Reference : tous les jours en tendance (sans touche), meme sortie temporelle.

Limites a garder en tete
------------------------
* Biais du survivant : l'univers est celui d'AUJOURD'HUI (S&P 500 + Nasdaq 100
  + Euronext actuels). Les titres sortis/faillis manquent : les resultats
  sont optimistes, surtout pour les rebonds « d'achat de creux ».
* Trades d'un meme jour tres correles (krach 2020, 2022) : les stats
  agregees ne valent pas autant que n observations independantes.

Utilisation
-----------
    python backtest_mm50_swing.py                      # US + Euronext, 10 ans
    python backtest_mm50_swing.py --univers us --echantillon 100
    python backtest_mm50_swing.py --stops 3 5 8 --rr 2 --max-hold 40
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
SELECTIONS = {"us": "us", "euronext": "euronext", "tout": "tout"}
HORIZONS = (10, 20, 40)
BUCKETS = [(1, "1er"), (2, "2e"), (3, "3e"), (4, "4e+")]


# ---------------------------------------------------------------------------
# Signaux
# ---------------------------------------------------------------------------

def detecter_touches(cadre, tol=0.005, reset=0.08, pente_jours=20):
    """
    Retourne un DataFrame indexe par la date du touche avec :
    n_touche, tendance (bool), confirme (bool), zone.
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


# ---------------------------------------------------------------------------
# Simulation d'un trade
# ---------------------------------------------------------------------------

def simuler(o, h, b, c, i_entree, prix, stop_pct, rr, max_hold):
    """
    Simule un long entre au prix `prix` a la seance i_entree (le stop et
    l'objectif sont testes des cette seance). Retourne (rendement %, R, jours, sortie).
    """
    stop = prix * (1 - stop_pct / 100)
    cible = prix * (1 + rr * stop_pct / 100)
    fin = min(i_entree + max_hold, len(c) - 1)
    for k in range(i_entree, fin + 1):
        # A la seance d'entree, l'ouverture est deja passee : pas de test de gap.
        if k > i_entree:
            if o[k] <= stop:
                return _res(o[k], prix, stop_pct, k - i_entree, "stop_gap")
            if o[k] >= cible:
                return _res(o[k], prix, stop_pct, k - i_entree, "cible_gap")
        if b[k] <= stop:
            return _res(stop, prix, stop_pct, k - i_entree, "stop")
        if h[k] >= cible:
            return _res(cible, prix, stop_pct, k - i_entree, "cible")
    if fin - i_entree < max_hold:  # donnees insuffisantes pour aller au bout
        return None
    return _res(c[fin], prix, stop_pct, fin - i_entree, "temps")


def _res(sortie, prix, stop_pct, jours, motif):
    ret = (sortie / prix - 1) * 100
    return ret, ret / stop_pct, jours, motif


def rendements_termes(c, i_ref, prix):
    """Rendements a HORIZONS seances (cloture) depuis l'entree, NaN si hors donnees."""
    sortie = {}
    for hz in HORIZONS:
        j = i_ref + hz
        sortie[f"r{hz}"] = (c[j] / prix - 1) * 100 if j < len(c) else np.nan
    return sortie


# ---------------------------------------------------------------------------
# Backtest d'un ticker
# ---------------------------------------------------------------------------

def trades_ticker(ticker, place, cadre, debut, stops, rr, max_hold, tol, reset):
    o = cadre["Open"].to_numpy(float)
    h = cadre["High"].to_numpy(float)
    b = cadre["Low"].to_numpy(float)
    c = cadre["Close"].to_numpy(float)
    dates = cadre.index

    touches = detecter_touches(cadre, tol=tol, reset=reset)
    lignes = []

    for t in touches.itertuples(index=False):
        if t.date < debut:
            continue
        i = t.i

        # --- entree limit, le jour du touche
        prix_limit = min(o[i], t.zone)
        # --- entree apres confirmation, ouverture du lendemain
        entrees = [("limit", i, prix_limit, True)]
        if t.confirme and i + 1 < len(c):
            entrees.append(("confirme", i + 1, o[i + 1], True))

        for variante, i_entree, prix, _ in entrees:
            if not np.isfinite(prix) or prix <= 0:
                continue
            base = {
                "ticker": ticker, "place": place, "date": dates[i],
                "n_touche": t.n_touche, "tendance": t.tendance,
                "variante": variante, "prix_entree": prix,
            }
            base.update(rendements_termes(c, i_entree, prix))
            for stop_pct in stops:
                r = simuler(o, h, b, c, i_entree, prix, stop_pct, rr, max_hold)
                if r is None:
                    continue
                lignes.append({**base, "stop_pct": stop_pct,
                               "ret": r[0], "R": r[1], "jours": r[2], "sortie": r[3]})
    return lignes


def reference_ticker(ticker, place, cadre, debut, pente_jours=20):
    """Tous les jours en tendance : rendements a HORIZONS seances depuis l'ouverture suivante."""
    c = cadre["Close"].to_numpy(float)
    o = cadre["Open"].to_numpy(float)
    ma200 = cadre["Close"].rolling(200).mean().to_numpy()
    dates = cadre.index
    lignes = []
    for t in range(200 + pente_jours + 1, len(c) - 1):
        if dates[t] < debut:
            continue
        if not (c[t - 1] > ma200[t - 1] and ma200[t - 1] > ma200[t - 1 - pente_jours]):
            continue
        prix = o[t + 1]
        if not np.isfinite(prix) or prix <= 0:
            continue
        ligne = {"ticker": ticker, "place": place, "date": dates[t]}
        ligne.update(rendements_termes(c, t + 1, prix))
        lignes.append(ligne)
    return lignes


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def bucket(n):
    return "4e+" if n >= 4 else {1: "1er", 2: "2e", 3: "3e"}[n]


def stats_trades(df):
    if df.empty:
        return {"n": 0}
    gains = df.loc[df["ret"] > 0, "ret"].sum()
    pertes = -df.loc[df["ret"] < 0, "ret"].sum()
    return {
        "n": len(df),
        "gagnants %": round((df["ret"] > 0).mean() * 100, 1),
        "ret moy %": round(df["ret"].mean(), 2),
        "R moy": round(df["R"].mean(), 2),
        "PF": round(gains / pertes, 2) if pertes > 0 else np.inf,
        "jours moy": round(df["jours"].mean(), 1),
    }


def tableau(df, par):
    lignes = []
    for cle, groupe in df.groupby(par, sort=False):
        cle = cle if isinstance(cle, tuple) else (cle,)
        lignes.append({**dict(zip(par, cle)), **stats_trades(groupe)})
    return pd.DataFrame(lignes)


def rapport(trades, ref, args, univers_info):
    sortie = []
    ajouter = sortie.append

    def md(df):
        return df.to_markdown(index=False) if len(df) else "_aucune donnee_"

    ajouter("# Backtest MM50 daily — rebond en tendance (MM200 ascendante)\n")
    ajouter(f"- Fenetre : {args.annees} ans · univers : {univers_info}")
    ajouter(f"- Zone de touche : MM50 + {args.tol * 100:.1f} % · reset du compteur : "
            f"cloture > MM50 + {args.reset * 100:.0f} %")
    ajouter(f"- Sortie : stop -X %, objectif +{args.rr:g}R, sortie temps {args.max_hold} seances · sans frais\n")

    tend = trades[trades["tendance"]].copy()
    tend["touche"] = tend["n_touche"].map(bucket)

    ajouter("## 1. Rendement brut a horizon fixe (sans stop), touches en tendance vs reference\n")
    for variante in ("limit", "confirme"):
        sous = tend[tend["variante"] == variante]
        ajouter(f"### Entree `{variante}`")
        lignes = []
        for _, nom in BUCKETS:
            g = sous[sous["touche"] == nom]
            for hz in HORIZONS:
                col = g[f"r{hz}"].dropna()
                if len(col):
                    lignes.append({"touche": nom, "horizon": f"{hz}j", "n": len(col),
                                   "ret moy %": round(col.mean(), 2),
                                   "mediane %": round(col.median(), 2),
                                   "% positifs": round((col > 0).mean() * 100, 1)})
        ajouter(md(pd.DataFrame(lignes)) + "\n")

    lignes = []
    for hz in HORIZONS:
        col = ref[f"r{hz}"].dropna()
        lignes.append({"reference": "tous jours en tendance", "horizon": f"{hz}j", "n": len(col),
                       "ret moy %": round(col.mean(), 2), "mediane %": round(col.median(), 2),
                       "% positifs": round((col > 0).mean() * 100, 1)})
    ajouter("### Reference (n'importe quel jour en tendance)")
    ajouter(md(pd.DataFrame(lignes)) + "\n")

    ajouter("## 2. Avec stop / objectif / sortie temps (par n° de touche)\n")
    for variante in ("limit", "confirme"):
        sous = tend[tend["variante"] == variante]
        ajouter(f"### Entree `{variante}`")
        t = tableau(sous, ["stop_pct", "touche"])
        if len(t):
            t["ordre"] = t["touche"].map({n: k for k, (_, n) in enumerate(BUCKETS)})
            t = t.sort_values(["stop_pct", "ordre"]).drop(columns="ordre")
        ajouter(md(t) + "\n")

    ajouter("## 3. Le n° de touche compte-t-il ? (1er-2e vs 3e+, stop median)\n")
    stop_med = sorted(args.stops)[len(args.stops) // 2]
    comp = tend[tend["stop_pct"] == stop_med].copy()
    comp["groupe"] = np.where(comp["n_touche"] <= 2, "1er-2e", "3e+")
    ajouter(f"Stop {stop_med} %")
    ajouter(md(tableau(comp, ["variante", "groupe"])) + "\n")

    ajouter("## 4. Sans filtre de tendance (memes regles, tous les touches)\n")
    tout = trades[trades["stop_pct"] == stop_med].copy()
    tout["groupe"] = np.where(tout["tendance"], "en tendance", "hors tendance")
    ajouter(md(tableau(tout, ["variante", "groupe"])) + "\n")

    ajouter("## 5. Robustesse par annee (entree `limit`, 1er-2e touche, en tendance)\n")
    an = tend[(tend["variante"] == "limit") & (tend["stop_pct"] == stop_med) & (tend["n_touche"] <= 2)].copy()
    an["annee"] = an["date"].dt.year
    ajouter(md(tableau(an, ["annee"])) + "\n")

    ajouter("## 6. Par place (entree `limit`, 1er-2e touche, en tendance)\n")
    ajouter(md(tableau(an, ["place"])) + "\n")

    ajouter("## Lecture\n")
    ajouter("- Compare les rendements a horizon fixe a la reference : sans ecart, le touche n'apporte rien.")
    ajouter("- `R moy` = rendement moyen / risque (stop). > 0 = esperance positive avant frais.")
    ajouter("- Biais du survivant : resultats optimistes. Un edge modeste (< 0,1 R) ne tient probablement pas en reel.")
    ajouter("- Verifie la colonne annee : un edge qui ne vit que sur 2020-2021 n'en est pas un.")
    return "\n".join(sortie)


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Backtest MM50 daily en tendance")
    p.add_argument("--univers", default="tout", choices=list(SELECTIONS))
    p.add_argument("--annees", type=int, default=10)
    p.add_argument("--stops", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--max-hold", type=int, default=40)
    p.add_argument("--tol", type=float, default=0.005)
    p.add_argument("--reset", type=float, default=0.08)
    p.add_argument("--echantillon", type=int, default=0,
                   help="limite le nombre de tickers par place (test rapide)")
    p.add_argument("--sortie", default="backtest_out")
    args = p.parse_args()

    import stromboli as sb  # import tardif : les fonctions ci-dessus restent testables sans

    debut = pd.Timestamp(datetime.now(timezone.utc).date()) - pd.DateOffset(years=args.annees)
    periode = f"{args.annees + 1}y"   # 1 an de plus pour chauffer la MM200
    sb.TWELVEDATA_API_KEY = ""        # repli Twelve Data limite a 700 jours : inutile ici

    univers = sb.construire_univers(args.univers)
    tous, refs, n_ok, n_susp, n_court = [], [], 0, 0, 0

    for place, tickers in univers.items():
        if place in ("Crypto", "Indices"):
            continue
        if args.echantillon:
            tickers = tickers[:args.echantillon]
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})", flush=True)
        donnees = sb.telecharger(tickers, periode)
        for ticker, cadre in donnees.items():
            if len(cadre) < 260:
                n_court += 1
                continue
            if sb._donnee_suspecte(cadre):
                n_susp += 1
                continue
            n_ok += 1
            tous += trades_ticker(ticker, place, cadre, debut, args.stops, args.rr,
                                  args.max_hold, args.tol, args.reset)
            refs += reference_ticker(ticker, place, cadre, debut)

    if not tous:
        print("Aucun trade : verifie EODHD_API_KEY et l'univers.")
        return 1

    trades = pd.DataFrame(tous)
    ref = pd.DataFrame(refs)
    info = f"{n_ok} titres exploitables ({n_court} trop courts, {n_susp} donnees suspectes ecartees)"
    texte = rapport(trades, ref, args, info)
    print("\n" + texte)

    sortie = Path(args.sortie)
    sortie.mkdir(exist_ok=True)
    trades.to_csv(sortie / "trades.csv", index=False)
    (sortie / "rapport.md").write_text(texte, encoding="utf-8")

    resume = os.getenv("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a", encoding="utf-8") as f:
            f.write(texte + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
