#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stromboli.py — Bot d'alerte Stromboli (methode Inchi)

Detecte les figures Stromboli en Heikin Ashi, sur les unites de temps
Daily et Weekly, sur les actions US (Nasdaq 100 + S&P 500) et Euronext
(Paris, Amsterdam, Bruxelles). Envoie les alertes sur Telegram.

Definition du Stromboli
-----------------------
Haussier  : >= 3 bougies HA rouges PLEINES consecutives (aucune meche haute,
            HA_high == HA_open) suivies IMMEDIATEMENT d'un doji.
Baissier  : >= 3 bougies HA vertes PLEINES consecutives (aucune meche basse,
            HA_low == HA_open) suivies IMMEDIATEMENT d'un doji.

Doji : corps <= SEUIL_DOJI % du range de la bougie, avec des meches des deux cotes.

Le bot signale la figure. Les invalidations et les take-profit sont geres
manuellement par l'operateur.

Utilisation
-----------
    python stromboli.py                      # scan D + W, envoi Telegram
    python stromboli.py --tf D               # Daily uniquement
    python stromboli.py --tf W               # Weekly uniquement
    python stromboli.py --dry-run            # affichage console, pas d'envoi
    python stromboli.py --univers us         # restreint l'univers
    python stromboli.py --valider-univers    # teste quels tickers repondent
    python stromboli.py --historique 3       # comptage des signaux sur 3 ans

Variables d'environnement
-------------------------
    TELEGRAM_BOT_TOKEN   (obligatoire sauf en --dry-run)
    TELEGRAM_CHAT_ID     (obligatoire sauf en --dry-run)
    STROMBOLI_MIN_BOUGIES   defaut 3
    STROMBOLI_SEUIL_DOJI    defaut 0.05
    STROMBOLI_TOLERANCE     defaut 0.0   (fraction du range toleree sur la meche)
"""

import argparse
import io
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------

RACINE = Path(__file__).resolve().parent

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

MIN_BOUGIES = int(os.getenv("STROMBOLI_MIN_BOUGIES", "3"))
SEUIL_DOJI = float(os.getenv("STROMBOLI_SEUIL_DOJI", "0.05"))
TOLERANCE_MECHE = float(os.getenv("STROMBOLI_TOLERANCE", "0.0"))

EPS = 1e-9  # marge anti-erreur d'arrondi flottant, pas une tolerance metier

TAILLE_LOT = 40          # tickers par requete yfinance
PERIODE_DAILY = "2y"     # historique telecharge
PERIODE_HISTORIQUE = "5y"

FICHIERS_EURONEXT = {
    "paris": "tickers_paris.txt",
    "amsterdam": "tickers_amsterdam.txt",
    "bruxelles": "tickers_bruxelles.txt",
}

UA = {"User-Agent": "Mozilla/5.0 (compatible; stromboli-bot/1.0)"}


# ---------------------------------------------------------------------------
# Univers
# ---------------------------------------------------------------------------

def _wikipedia_table(url, colonne):
    """Recupere une colonne de tickers depuis une table Wikipedia."""
    reponse = requests.get(url, headers=UA, timeout=30)
    reponse.raise_for_status()
    tables = pd.read_html(io.StringIO(reponse.text))
    for table in tables:
        if colonne in table.columns:
            valeurs = table[colonne].dropna().astype(str).tolist()
            return [v.strip().replace(".", "-").upper() for v in valeurs if v.strip()]
    return []


def univers_us():
    """Nasdaq 100 + S&P 500 depuis Wikipedia, dedoublonne."""
    tickers = []
    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Ticker"),
        ("https://en.wikipedia.org/wiki/Nasdaq-100", "Symbol"),
    ]
    for url, colonne in sources:
        try:
            trouves = _wikipedia_table(url, colonne)
            if trouves:
                tickers.extend(trouves)
                print(f"  {len(trouves)} tickers depuis {url.split('/')[-1]} [{colonne}]")
        except Exception as erreur:
            print(f"  echec {url.split('/')[-1]}: {erreur}")

    if not tickers:
        secours = RACINE / "tickers_us.txt"
        if secours.exists():
            print("  bascule sur le fichier de secours tickers_us.txt")
            tickers = charger_fichier(secours)

    return sorted(set(tickers))


def charger_fichier(chemin):
    """Lit un fichier de tickers : une ligne par ticker, # pour commenter."""
    chemin = Path(chemin)
    if not chemin.exists():
        print(f"  fichier absent : {chemin.name}")
        return []
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    tickers = []
    for ligne in lignes:
        ligne = ligne.split("#")[0].strip()
        if ligne:
            tickers.append(ligne.upper())
    return tickers


def construire_univers(selection):
    """selection : 'tout', 'us', 'euronext', 'paris', 'amsterdam', 'bruxelles'."""
    univers = {}

    if selection in ("tout", "us"):
        print("Univers US :")
        univers["US"] = univers_us()

    for place, fichier in FICHIERS_EURONEXT.items():
        if selection in ("tout", "euronext", place):
            tickers = charger_fichier(RACINE / fichier)
            if tickers:
                print(f"Univers {place.capitalize()} : {len(tickers)} tickers")
                univers[place.capitalize()] = tickers

    total = sum(len(v) for v in univers.values())
    print(f"Total : {total} tickers\n")
    return univers


# ---------------------------------------------------------------------------
# Heikin Ashi
# ---------------------------------------------------------------------------

def heikin_ashi(ohlc):
    """
    Convertit un DataFrame OHLC classique en bougies Heikin Ashi.

    HA_close = (O + H + L + C) / 4
    HA_open  = (HA_open precedent + HA_close precedent) / 2
    HA_high  = max(H, HA_open, HA_close)
    HA_low   = min(L, HA_open, HA_close)
    """
    ouverture = ohlc["Open"].to_numpy(dtype=float)
    haut = ohlc["High"].to_numpy(dtype=float)
    bas = ohlc["Low"].to_numpy(dtype=float)
    cloture = ohlc["Close"].to_numpy(dtype=float)

    ha_close = (ouverture + haut + bas + cloture) / 4.0

    ha_open = np.empty(len(ohlc), dtype=float)
    ha_open[0] = (ouverture[0] + cloture[0]) / 2.0
    for i in range(1, len(ohlc)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([haut, ha_open, ha_close])
    ha_low = np.minimum.reduce([bas, ha_open, ha_close])

    resultat = pd.DataFrame(
        {"open": ha_open, "high": ha_high, "low": ha_low, "close": ha_close},
        index=ohlc.index,
    )
    if "Volume" in ohlc.columns:
        resultat["volume"] = ohlc["Volume"].to_numpy(dtype=float)
    return resultat


def to_weekly(ohlc):
    """
    Agrege en bougies hebdomadaires (semaine calendaire, cloture vendredi).
    La semaine en cours, incomplete, est retiree.
    """
    hebdo = ohlc.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "Close"])

    if len(hebdo) == 0:
        return hebdo

    # Le label de la derniere ligne est le vendredi de la semaine. Si la derniere
    # bougie daily est anterieure a ce vendredi, la semaine n'est pas terminee.
    dernier_jour = ohlc.index[-1]
    if hebdo.index[-1] > dernier_jour:
        hebdo = hebdo.iloc[:-1]

    return hebdo


# ---------------------------------------------------------------------------
# Detection du Stromboli
# ---------------------------------------------------------------------------

def _range(ha, i):
    return float(ha["high"].iloc[i] - ha["low"].iloc[i])


def est_rouge_pleine(ha, i):
    """Bougie HA rouge sans meche haute : HA_high == HA_open."""
    ouverture = float(ha["open"].iloc[i])
    cloture = float(ha["close"].iloc[i])
    haut = float(ha["high"].iloc[i])
    if cloture >= ouverture:
        return False
    etendue = _range(ha, i)
    if etendue <= 0:
        return False
    return (haut - ouverture) <= TOLERANCE_MECHE * etendue + EPS


def est_verte_pleine(ha, i):
    """Bougie HA verte sans meche basse : HA_low == HA_open."""
    ouverture = float(ha["open"].iloc[i])
    cloture = float(ha["close"].iloc[i])
    bas = float(ha["low"].iloc[i])
    if cloture <= ouverture:
        return False
    etendue = _range(ha, i)
    if etendue <= 0:
        return False
    return (ouverture - bas) <= TOLERANCE_MECHE * etendue + EPS


def est_doji(ha, i):
    """Petit corps, avec des meches des deux cotes."""
    ouverture = float(ha["open"].iloc[i])
    cloture = float(ha["close"].iloc[i])
    haut = float(ha["high"].iloc[i])
    bas = float(ha["low"].iloc[i])

    etendue = haut - bas
    if etendue <= 0:
        return False

    corps = abs(cloture - ouverture)
    if corps > SEUIL_DOJI * etendue:
        return False

    meche_haute = haut - max(ouverture, cloture)
    meche_basse = min(ouverture, cloture) - bas
    return meche_haute > EPS and meche_basse > EPS


def compter_serie(ha, fin, test):
    """Nombre de bougies consecutives verifiant `test` en remontant depuis `fin`."""
    compte = 0
    i = fin
    while i >= 0 and test(ha, i):
        compte += 1
        i -= 1
    return compte


def detecter_stromboli(ha, i):
    """
    Teste si la bougie d'indice i est le doji d'un Stromboli.
    Retourne un dict ou None.
    """
    if i < MIN_BOUGIES:
        return None
    if not est_doji(ha, i):
        return None

    serie_rouge = compter_serie(ha, i - 1, est_rouge_pleine)
    if serie_rouge >= MIN_BOUGIES:
        sens = "haussier"
        longueur = serie_rouge
    else:
        serie_verte = compter_serie(ha, i - 1, est_verte_pleine)
        if serie_verte >= MIN_BOUGIES:
            sens = "baissier"
            longueur = serie_verte
        else:
            return None

    etendue = _range(ha, i)
    corps = abs(float(ha["close"].iloc[i]) - float(ha["open"].iloc[i]))

    resultat = {
        "sens": sens,
        "bougies": longueur,
        "date": ha.index[i],
        "ha_close": float(ha["close"].iloc[i]),
        "ratio_corps": corps / etendue if etendue > 0 else 0.0,
    }

    if "volume" in ha.columns:
        volume = float(ha["volume"].iloc[i])
        debut = max(0, i - 20)
        moyenne = float(ha["volume"].iloc[debut:i].mean()) if i > debut else 0.0
        resultat["volume"] = volume
        resultat["volume_ratio"] = volume / moyenne if moyenne > 0 else None

    return resultat


# ---------------------------------------------------------------------------
# Telechargement
# ---------------------------------------------------------------------------

def telecharger(tickers, periode):
    """Telecharge en lots. Retourne {ticker: DataFrame OHLCV}."""
    donnees = {}
    lots = [tickers[i:i + TAILLE_LOT] for i in range(0, len(tickers), TAILLE_LOT)]

    for numero, lot in enumerate(lots, 1):
        print(f"  lot {numero}/{len(lots)} ({len(lot)} tickers)...", flush=True)
        try:
            brut = yf.download(
                lot,
                period=periode,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
        except Exception as erreur:
            print(f"    echec du lot : {erreur}")
            continue

        for ticker in lot:
            try:
                if len(lot) == 1:
                    cadre = brut
                else:
                    if ticker not in brut.columns.get_level_values(0):
                        continue
                    cadre = brut[ticker]
                cadre = cadre.dropna(subset=["Open", "High", "Low", "Close"])
                if len(cadre) >= MIN_BOUGIES + 25:
                    donnees[ticker] = cadre
            except Exception:
                continue

        time.sleep(0.4)

    return donnees


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scanner(univers, timeframes, periode=PERIODE_DAILY):
    """Scan de la derniere bougie cloturee. Retourne la liste des signaux."""
    signaux = []

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers")
        donnees = telecharger(tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, ohlc in donnees.items():
            for tf in timeframes:
                cadre = ohlc if tf == "D" else to_weekly(ohlc)
                if len(cadre) < MIN_BOUGIES + 2:
                    continue

                ha = heikin_ashi(cadre)
                trouve = detecter_stromboli(ha, len(ha) - 1)
                if trouve:
                    trouve["ticker"] = ticker
                    trouve["place"] = place
                    trouve["tf"] = tf
                    signaux.append(trouve)
                    print(f"  >> STROMBOLI {tf} {trouve['sens']} : {ticker}")

    return signaux


def scanner_historique(univers, timeframes, annees):
    """Compte tous les Stromboli de l'historique. Pour calibrer les seuils."""
    lignes = []
    periode = f"{annees}y"

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger(tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, ohlc in donnees.items():
            for tf in timeframes:
                cadre = ohlc if tf == "D" else to_weekly(ohlc)
                if len(cadre) < MIN_BOUGIES + 2:
                    continue
                ha = heikin_ashi(cadre)
                for i in range(MIN_BOUGIES, len(ha)):
                    trouve = detecter_stromboli(ha, i)
                    if trouve:
                        trouve["ticker"] = ticker
                        trouve["place"] = place
                        trouve["tf"] = tf
                        lignes.append(trouve)

    return lignes


def valider_univers(univers):
    """Identifie les tickers qui ne renvoient pas de donnees."""
    morts = {}
    for place, tickers in univers.items():
        print(f"\n[{place}] validation de {len(tickers)} tickers")
        donnees = telecharger(tickers, "3mo")
        absents = sorted(set(tickers) - set(donnees.keys()))
        morts[place] = absents
        print(f"  {len(donnees)} OK, {len(absents)} sans donnees")
        for ticker in absents:
            print(f"    - {ticker}")
    return morts


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def envoyer_telegram(texte, dry_run=False):
    if dry_run:
        print("\n--- message Telegram (dry-run) ---")
        print(texte)
        print("--- fin ---\n")
        return True

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID absents, envoi ignore.")
        print(texte)
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for morceau in decouper(texte, 3800):
        try:
            reponse = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": morceau,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )
            if reponse.status_code != 200:
                print(f"Telegram HTTP {reponse.status_code} : {reponse.text[:200]}")
                return False
        except Exception as erreur:
            print(f"Telegram erreur : {erreur}")
            return False
        time.sleep(0.5)
    return True


def decouper(texte, taille):
    if len(texte) <= taille:
        return [texte]
    morceaux, courant = [], ""
    for ligne in texte.split("\n"):
        if len(courant) + len(ligne) + 1 > taille:
            morceaux.append(courant)
            courant = ligne
        else:
            courant = f"{courant}\n{ligne}" if courant else ligne
    if courant:
        morceaux.append(courant)
    return morceaux


def formater(signaux, timeframes):
    horodatage = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    if not signaux:
        return f"<b>STROMBOLI</b> — {horodatage}\n\nAucun signal sur {', '.join(timeframes)}."

    lignes = [f"<b>STROMBOLI</b> — {horodatage}", ""]

    for tf in timeframes:
        for sens in ("haussier", "baissier"):
            groupe = [s for s in signaux if s["tf"] == tf and s["sens"] == sens]
            if not groupe:
                continue

            etiquette = "DAILY" if tf == "D" else "WEEKLY"
            fleche = "▲" if sens == "haussier" else "▼"
            lignes.append(f"<b>{fleche} {etiquette} {sens.upper()}</b> ({len(groupe)})")

            for signal in sorted(groupe, key=lambda s: s["ticker"]):
                date = signal["date"].strftime("%d/%m")
                detail = (
                    f"corps {signal['ratio_corps'] * 100:.1f}% · "
                    f"{signal['bougies']} bougies"
                )
                ratio = signal.get("volume_ratio")
                if ratio:
                    detail += f" · vol x{ratio:.1f}"
                lignes.append(
                    f"  <code>{signal['ticker']}</code> — {signal['ha_close']:.2f} "
                    f"({date})\n     {detail}"
                )
            lignes.append("")

    lignes.append(f"<i>Invalidations et TP a gerer manuellement.</i>")
    return "\n".join(lignes)


def resume_historique(lignes, annees):
    if not lignes:
        return "Aucun Stromboli sur la periode."

    cadre = pd.DataFrame(lignes)
    cadre["annee"] = pd.to_datetime(cadre["date"]).dt.year

    sortie = [f"Comptage des Stromboli sur {annees} ans", ""]
    sortie.append(str(pd.crosstab([cadre["tf"], cadre["sens"]], cadre["annee"])))
    sortie.append("")
    sortie.append("Par place :")
    sortie.append(str(pd.crosstab([cadre["tf"], cadre["sens"]], cadre["place"])))
    sortie.append("")
    sortie.append(f"Total : {len(cadre)} signaux")
    sortie.append(
        f"Parametres : MIN_BOUGIES={MIN_BOUGIES} "
        f"SEUIL_DOJI={SEUIL_DOJI} TOLERANCE={TOLERANCE_MECHE}"
    )
    return "\n".join(sortie)


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------

def main():
    parseur = argparse.ArgumentParser(description="Bot d'alerte Stromboli")
    parseur.add_argument("--tf", default="DW", help="D, W ou DW (defaut DW)")
    parseur.add_argument(
        "--univers",
        default="tout",
        choices=["tout", "us", "euronext", "paris", "amsterdam", "bruxelles"],
    )
    parseur.add_argument("--dry-run", action="store_true", help="pas d'envoi Telegram")
    parseur.add_argument("--valider-univers", action="store_true")
    parseur.add_argument("--historique", type=int, metavar="ANNEES")
    args = parseur.parse_args()

    timeframes = [c for c in "DW" if c in args.tf.upper()]
    if not timeframes:
        print("--tf doit contenir D et/ou W")
        return 1

    print("=" * 60)
    print("BOT STROMBOLI — methode Inchi")
    print(
        f"Parametres : {MIN_BOUGIES} bougies min · doji <= {SEUIL_DOJI * 100:.0f}% "
        f"· tolerance meche {TOLERANCE_MECHE * 100:.0f}%"
    )
    print("=" * 60 + "\n")

    univers = construire_univers(args.univers)
    if not univers:
        print("Univers vide.")
        return 1

    if args.valider_univers:
        valider_univers(univers)
        return 0

    if args.historique:
        lignes = scanner_historique(univers, timeframes, args.historique)
        rapport = resume_historique(lignes, args.historique)
        print("\n" + rapport)
        chemin = RACINE / "historique_stromboli.csv"
        if lignes:
            pd.DataFrame(lignes).to_csv(chemin, index=False)
            print(f"\nDetail ecrit dans {chemin.name}")
        return 0

    signaux = scanner(univers, timeframes)
    print(f"\n{len(signaux)} signal(aux) detecte(s)")

    envoyer_telegram(formater(signaux, timeframes), dry_run=args.dry_run)

    resume = os.getenv("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a", encoding="utf-8") as fichier:
            fichier.write(f"## Stromboli — {len(signaux)} signal(aux)\n\n")
            for signal in signaux:
                fichier.write(
                    f"- `{signal['ticker']}` {signal['tf']} {signal['sens']} "
                    f"({signal['bougies']} bougies)\n"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
