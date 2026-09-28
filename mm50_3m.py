#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm50_3m.py — Alertes MM50 trimestrielle (bougies de 3 mois)

Surveille les actions US (S&P 500 + Nasdaq 100, qui incluent les 30 valeurs
du Dow Jones) et Euronext (Paris, Amsterdam, Bruxelles), et previent quand
un titre revient PAR LE HAUT vers sa moyenne mobile 50 sur bougies
TRIMESTRIELLES.

Alertes (une seule fois chacune, pas de retrigger)
--------------------------------------------------
Alerte 1 — approche : cloture a moins de 6 % au-dessus de la MM50.
Alerte 2 — contact  : le plus bas de la seance descend jusqu'a MM50 + 0,5 %.
           Declenchee quelle que soit la cloture : rebond au-dessus de la
           zone, cloture dans la zone, ou cassure sous la MM50 (le message
           precise lequel).

Memoire : mm50_3m_state.json, commite par le workflow. Un titre ne
redevient alertable qu'apres s'etre eloigne a plus de 10 % au-dessus de la
MM50. Un titre inconnu (premier lancement, nouvel entrant dans un indice,
titre qui atteint 50 trimestres d'historique) est enregistre sans alerte.

Calcul
------
MM50 = moyenne des clotures des 50 derniers trimestres calendaires
(T1 = janv-mars, etc.), trimestre en cours inclus avec la cloture du jour,
comme TradingView en unite 3M. Il faut donc au moins 50 trimestres
d'historique : les titres plus recents sont ignores, la crypto est exclue.

Reutilise stromboli.py (univers, telechargement EODHD, Telegram) sans le
modifier.

Utilisation
-----------
    python mm50_3m.py                    # scan US + Euronext, envoi Telegram
    python mm50_3m.py --univers us       # US uniquement
    python mm50_3m.py --dry-run          # console seulement, etat NON sauvegarde
    python mm50_3m.py --diagnostic AMZN  # detail du calcul, a comparer a TradingView
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import stromboli as sb

# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------

RACINE = Path(__file__).resolve().parent
FICHIER_ETAT = RACINE / "mm50_3m_state.json"

NB_TRIMESTRES = 50
SEUIL_APPROCHE_PCT = float(os.getenv("MM50_3M_APPROCHE", "6.0"))
ZONE_CONTACT_PCT = float(os.getenv("MM50_3M_ZONE", "0.5"))
SEUIL_RESET_PCT = float(os.getenv("MM50_3M_RESET", "10.0"))

PERIODE = "14y"            # ~56 trimestres, marge au-dessus des 50 requis
FRAICHEUR_MAX_JOURS = 5    # derniere seance trop ancienne = titre suspendu/radie

LOIN, APPROCHE, CONTACT = "loin", "approche", "contact"
SELECTIONS = {"us": ["us"], "euronext": ["euronext"], "tout": ["us", "euronext"]}

# Le repli Twelve Data de stromboli.telecharger() est limite a 700 jours
# (inutile pour 50 trimestres) et attend 8 s par ticker manquant : on le
# coupe ici, sans toucher a Stromboli.
sb.TWELVEDATA_API_KEY = ""


# ---------------------------------------------------------------------------
# Calcul
# ---------------------------------------------------------------------------

def to_quarterly(ohlc):
    """
    Agrege en bougies trimestrielles calendaires (fin mars/juin/sept/dec).
    Le trimestre EN COURS est conserve : sa cloture est celle du jour,
    exactement comme la bougie 3M en cours sur TradingView.
    """
    return ohlc.resample("QE-DEC").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna(subset=["Close"])


def mesurer(cadre):
    """
    Calcule la MM50 trimestrielle et la position de la derniere seance.
    Retourne None si moins de NB_TRIMESTRES trimestres d'historique.
    """
    trimestres = to_quarterly(cadre)
    if len(trimestres) < NB_TRIMESTRES:
        return None

    mm50 = float(trimestres["Close"].iloc[-NB_TRIMESTRES:].mean())
    if mm50 <= 0:
        return None

    close = float(cadre["Close"].iloc[-1])
    bas = float(cadre["Low"].iloc[-1])
    return {
        "date": cadre.index[-1],
        "close": close,
        "bas": bas,
        "mm50": mm50,
        "dist_close": (close / mm50 - 1) * 100,
        "dist_bas": (bas / mm50 - 1) * 100,
        "trimestres": len(trimestres),
    }


def evaluer(etat, m):
    """
    Machine a etats d'un titre. etat : None (inconnu), LOIN, APPROCHE, CONTACT.
    Retourne (nouvel_etat, alerte) avec alerte dans (None, APPROCHE, CONTACT).

    Le contact est teste en premier : un titre qui touche la zone puis
    rebondit fort dans la meme seance doit quand meme alerter.
    """
    contact = m["dist_bas"] <= ZONE_CONTACT_PCT

    if etat is None:  # titre inconnu : enregistre sans alerte
        if contact:
            return CONTACT, None
        if m["dist_close"] < SEUIL_APPROCHE_PCT:
            return APPROCHE, None
        return LOIN, None

    if contact:
        return (CONTACT, None) if etat == CONTACT else (CONTACT, CONTACT)

    if m["dist_close"] > SEUIL_RESET_PCT:
        return LOIN, None

    if etat == LOIN and m["dist_close"] < SEUIL_APPROCHE_PCT:
        return APPROCHE, APPROCHE

    return etat, None


def libelle_contact(m):
    """Ce que la cloture a fait apres le contact avec la zone."""
    d = m["dist_close"]
    if d > ZONE_CONTACT_PCT:
        return "rebond, clôture au-dessus de la zone"
    if d >= -ZONE_CONTACT_PCT:
        return "clôture dans la zone"
    return "cassure, clôture sous la MM50"


# ---------------------------------------------------------------------------
# Etat
# ---------------------------------------------------------------------------

def charger_etat():
    if FICHIER_ETAT.exists():
        try:
            etat = json.loads(FICHIER_ETAT.read_text(encoding="utf-8"))
            etat.setdefault("tickers", {})
            return etat
        except Exception as erreur:
            print(f"Etat illisible ({erreur}), redemarrage a vide.")
    return {"tickers": {}}


def sauver_etat(etat):
    etat["maj"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    FICHIER_ETAT.write_text(
        json.dumps(etat, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Messages Telegram
# ---------------------------------------------------------------------------

def _ligne(r):
    m = r["mesure"]
    return (
        f"  <code>{r['ticker']}</code> [{r['place']}] — "
        f"{m['close']:.2f} · MM50 {m['mm50']:.2f}"
    )


def _date_seance(resultats):
    dates = [r["mesure"]["date"] for r in resultats]
    return max(dates).strftime("%d/%m/%Y") if dates else "?"


def formater_alertes(contacts, approches):
    lignes = [f"<b>MM50 TRIMESTRIELLE</b> — séance du {_date_seance(contacts + approches)}", ""]

    if contacts:
        lignes.append(f"<b>🔔 CONTACT MM50</b> ({len(contacts)})")
        for r in sorted(contacts, key=lambda r: r["mesure"]["dist_close"]):
            m = r["mesure"]
            lignes.append(_ligne(r))
            lignes.append(
                f"     plus bas {m['bas']:.2f} ({m['dist_bas']:+.1f} %) · "
                f"clôture {m['dist_close']:+.1f} % → {libelle_contact(m)}"
            )
        lignes.append("")

    if approches:
        lignes.append(f"<b>👀 APPROCHE &lt; {SEUIL_APPROCHE_PCT:.0f} %</b> ({len(approches)})")
        for r in sorted(approches, key=lambda r: r["mesure"]["dist_close"]):
            lignes.append(f"{_ligne(r)} · {r['mesure']['dist_close']:+.1f} %")
        lignes.append("")

    lignes.append(
        f"<i>MM50 sur bougies de 3 mois · une alerte par étape, "
        f"réarmée au-delà de +{SEUIL_RESET_PCT:.0f} %.</i>"
    )
    return "\n".join(lignes)


def formater_initial(resultats, trop_courts):
    contacts = [r for r in resultats
                if r["etat"] == CONTACT and r["mesure"]["dist_close"] >= -ZONE_CONTACT_PCT]
    sous = [r for r in resultats
            if r["etat"] == CONTACT and r["mesure"]["dist_close"] < -ZONE_CONTACT_PCT]
    approches = [r for r in resultats if r["etat"] == APPROCHE]

    lignes = [
        f"<b>MM50 TRIMESTRIELLE — situation initiale</b> ({_date_seance(resultats)})",
        f"{len(resultats)} titres suivis · {trop_courts} ignorés (moins de {NB_TRIMESTRES} trimestres)",
        "<i>Premier lancement : ces titres sont enregistrés comme déjà signalés, "
        "sans alerte. Les alertes normales démarrent au prochain run.</i>",
        "",
    ]
    if contacts:
        lignes.append(f"<b>Au contact de la MM50</b> ({len(contacts)})")
        for r in sorted(contacts, key=lambda r: r["mesure"]["dist_close"]):
            lignes.append(f"{_ligne(r)} · {r['mesure']['dist_close']:+.1f} %")
        lignes.append("")
    if approches:
        lignes.append(f"<b>En approche</b> ({len(approches)})")
        for r in sorted(approches, key=lambda r: r["mesure"]["dist_close"]):
            lignes.append(f"{_ligne(r)} · {r['mesure']['dist_close']:+.1f} %")
        lignes.append("")
    lignes.append(f"{len(sous)} titres déjà sous la MM50 (non listés).")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scanner(selection, etat):
    """
    Telecharge, mesure et met a jour l'etat (en memoire).
    Retourne (resultats, stats) ; chaque resultat porte son alerte eventuelle.
    """
    tickers_etat = etat["tickers"]
    resultats = []
    stats = {"trop_courts": 0, "perimes": 0, "nouveaux": 0}

    for cle in SELECTIONS[selection]:
        univers = sb.construire_univers(cle)
        for place, tickers in univers.items():
            print(f"\n[{place}] telechargement de {len(tickers)} tickers ({PERIODE})")
            donnees = sb.telecharger(tickers, PERIODE)
            print(f"  {len(donnees)} tickers recuperes")
            if not donnees:
                continue

            derniere = max(c.index[-1] for c in donnees.values())
            for ticker, cadre in donnees.items():
                if (derniere - cadre.index[-1]).days > FRAICHEUR_MAX_JOURS:
                    stats["perimes"] += 1
                    continue

                m = mesurer(cadre)
                if m is None:
                    stats["trop_courts"] += 1
                    continue

                precedent = tickers_etat.get(ticker, {}).get("etat")
                if precedent is None:
                    stats["nouveaux"] += 1
                nouvel, alerte = evaluer(precedent, m)

                entree = tickers_etat.get(ticker, {})
                if nouvel != precedent:
                    entree["depuis"] = m["date"].strftime("%Y-%m-%d")
                entree.update({
                    "etat": nouvel,
                    "place": place,
                    "mm50": round(m["mm50"], 4),
                    "dist": round(m["dist_close"], 2),
                })
                tickers_etat[ticker] = entree

                resultats.append({
                    "ticker": ticker, "place": place, "mesure": m,
                    "etat": nouvel, "alerte": alerte,
                })
                if alerte:
                    print(f"  >> {alerte.upper()} : {ticker} ({m['dist_close']:+.1f} %)")

    return resultats, stats


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def _cloture_splits_seulement(ticker):
    """
    Cloture ajustee des SPLITS uniquement (pas des dividendes), proche de
    l'affichage par defaut de TradingView. Deux appels EODHD : cours bruts
    + historique des splits. Sert seulement au diagnostic.
    """
    symbole = sb._symbole_eodhd(ticker)
    depuis = (datetime.now(timezone.utc) - pd.Timedelta(days=sb._periode_en_jours(PERIODE))).strftime("%Y-%m-%d")
    params = {"api_token": sb.EODHD_API_KEY, "fmt": "json", "from": depuis}

    brut = requests.get(f"{sb.EODHD_API}/eod/{symbole}", params={**params, "period": "d"}, timeout=30)
    brut.raise_for_status()
    df = pd.DataFrame(brut.json())
    df["date"] = pd.to_datetime(df["date"])
    close = df.set_index("date").sort_index()["close"].astype(float)

    splits = requests.get(f"{sb.EODHD_API}/splits/{symbole}", params=params, timeout=30)
    splits.raise_for_status()
    for split in splits.json():
        a, b = (float(x) for x in str(split["split"]).split("/"))
        if a > 0 and b > 0:
            close[close.index < pd.Timestamp(split["date"])] /= a / b
    return close


def diagnostiquer(ticker):
    print(f"Telechargement de {ticker} ({PERIODE})...")
    donnees = sb.telecharger([ticker], PERIODE)
    if ticker not in donnees:
        print(f"Aucune donnee pour {ticker}.")
        return

    cadre = donnees[ticker]
    trimestres = to_quarterly(cadre)
    print(f"\n{len(trimestres)} trimestres (premier : {trimestres.index[0].date()})")
    print("\nDerniers trimestres (cloture ajustee splits + dividendes) :")
    for date, ligne in trimestres.tail(8).iterrows():
        print(f"  {date.date()}  {ligne['Close']:>10.2f}")

    m = mesurer(cadre)
    if m is None:
        print(f"\nMoins de {NB_TRIMESTRES} trimestres : titre ignore par le bot.")
        return

    print(f"\nSeance du {m['date'].date()} : cloture {m['close']:.2f} · plus bas {m['bas']:.2f}")
    print(f"MM50 3M (ajustee dividendes, utilisee par le bot) : {m['mm50']:.2f}")
    print(f"  distance cloture {m['dist_close']:+.2f} % · plus bas {m['dist_bas']:+.2f} %")

    try:
        close_splits = _cloture_splits_seulement(ticker)
        trim_splits = close_splits.resample("QE-DEC").last().dropna()
        if len(trim_splits) >= NB_TRIMESTRES:
            mm50_splits = float(trim_splits.iloc[-NB_TRIMESTRES:].mean())
            ecart = (m["mm50"] / mm50_splits - 1) * 100
            print(f"MM50 3M (splits seulement, ~TradingView)       : {mm50_splits:.2f}")
            print(f"  ecart entre les deux : {ecart:+.2f} % (a comparer a la zone de ±{ZONE_CONTACT_PCT} %)")
    except Exception as erreur:
        print(f"MM50 splits seulement indisponible : {erreur}")

    etat = charger_etat()["tickers"].get(ticker)
    print(f"\nEtat enregistre : {etat if etat else 'aucun (titre inconnu)'}")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main():
    parseur = argparse.ArgumentParser(description="Alertes MM50 trimestrielle")
    parseur.add_argument("--univers", default="tout", choices=list(SELECTIONS))
    parseur.add_argument("--dry-run", action="store_true",
                         help="affichage console, pas d'envoi Telegram, etat non sauvegarde")
    parseur.add_argument("--diagnostic", metavar="TICKER",
                         help="detail du calcul pour un ticker (ex: AMZN, MC.PA)")
    args = parseur.parse_args()

    if args.diagnostic:
        diagnostiquer(args.diagnostic.upper())
        return 0

    print("=" * 60)
    print("BOT MM50 TRIMESTRIELLE")
    print(f"Approche < {SEUIL_APPROCHE_PCT} % · contact <= MM50 + {ZONE_CONTACT_PCT} % "
          f"· rearmement > {SEUIL_RESET_PCT} %")
    print("=" * 60)

    etat = charger_etat()
    premier_run = not etat["tickers"]

    resultats, stats = scanner(args.univers, etat)
    contacts = [r for r in resultats if r["alerte"] == CONTACT]
    approches = [r for r in resultats if r["alerte"] == APPROCHE]

    print(f"\n{len(resultats)} titres suivis · {stats['trop_courts']} trop courts · "
          f"{stats['perimes']} perimes · {stats['nouveaux']} nouveaux (enregistres sans alerte)")
    print(f"{len(contacts)} contact(s) · {len(approches)} approche(s)")

    if not resultats:
        print("Aucun titre exploitable, etat non modifie.")
        return 1

    if premier_run:
        sb.envoyer_telegram(formater_initial(resultats, stats["trop_courts"]), dry_run=args.dry_run)
    elif contacts or approches:
        sb.envoyer_telegram(formater_alertes(contacts, approches), dry_run=args.dry_run)
    else:
        print("Aucune alerte aujourd'hui, pas de message.")

    if args.dry_run:
        print("Dry-run : etat non sauvegarde.")
    else:
        sauver_etat(etat)
        print(f"Etat sauvegarde dans {FICHIER_ETAT.name}")

    resume = os.getenv("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a", encoding="utf-8") as fichier:
            fichier.write(f"## MM50 trimestrielle — {len(resultats)} titres suivis\n\n")
            fichier.write(f"- {stats['trop_courts']} ignores (< {NB_TRIMESTRES} trimestres), "
                          f"{stats['perimes']} perimes, {stats['nouveaux']} nouveaux\n")
            for r in contacts:
                fichier.write(f"- CONTACT `{r['ticker']}` ({r['mesure']['dist_close']:+.1f} %)\n")
            for r in approches:
                fichier.write(f"- APPROCHE `{r['ticker']}` ({r['mesure']['dist_close']:+.1f} %)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
