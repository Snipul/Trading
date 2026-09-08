#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stromboli.py — Bot d'alerte Stromboli (methode Inchi)

Detecte les figures Stromboli en Heikin Ashi, sur les unites de temps
Daily, Weekly et Monthly, sur les actions US (Nasdaq 100 + S&P 500) et Euronext
(Paris, Amsterdam, Bruxelles). Envoie les alertes sur Telegram.

Definition du Stromboli
-----------------------
Haussier (long, seul cote actif) : >= 3 bougies HA rouges PLEINES
            consecutives (aucune meche haute, HA_high == HA_open) suivies
            IMMEDIATEMENT d'un doji.
Le cote baissier (vert plein + doji) n'est pas detecte pour le moment,
retire volontairement (pas utilise cote trading). La fonction
est_verte_pleine() reste dans le code pour pouvoir le reactiver facilement.

Doji : corps <= SEUIL_DOJI % du range de la bougie, avec des meches des deux cotes.

Definition de Fernanda
-----------------------
Fernanda (long, seul cote actif) : apres un Stromboli haussier, cloture
            au-dessus de la M7 (ascendante) et de la Tenkan (9 periodes),
            sur HA. Fernando/short n'est pas detecte pour le meme motif.

Le Stromboli reste surveille tant qu'aucune bougie ne fait un plus bas (resp.
plus haut) HA inferieur (resp. superieur) a celui de la bougie precedente.
Recalcule integralement a chaque scan a partir des 2 ans d'historique
telecharges : aucun etat n'est stocke entre deux executions.

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

# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------

RACINE = Path(__file__).resolve().parent

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
EODHD_API = "https://eodhd.com/api"

MIN_BOUGIES = int(os.getenv("STROMBOLI_MIN_BOUGIES", "3"))
SEUIL_DOJI = float(os.getenv("STROMBOLI_SEUIL_DOJI", "0.05"))
TOLERANCE_MECHE = float(os.getenv("STROMBOLI_TOLERANCE", "0.0"))
INCLURE_DRAGONFLY = os.getenv("STROMBOLI_DOJI_DRAGONFLY", "false").strip().lower() in ("1", "true", "vrai", "oui")

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
KRAKEN_API = "https://api.kraken.com/0/public"


# ---------------------------------------------------------------------------
# Univers
# ---------------------------------------------------------------------------

def _csv_colonne(url, colonne):
    """Recupere une colonne de tickers depuis un CSV distant."""
    reponse = requests.get(url, headers=UA, timeout=30)
    reponse.raise_for_status()
    table = pd.read_csv(io.StringIO(reponse.text))
    if colonne not in table.columns:
        return []
    valeurs = table[colonne].dropna().astype(str).tolist()
    return [v.strip().replace(".", "-").upper() for v in valeurs if v.strip()]


def _wikipedia_table(url, colonne):
    """Recupere une colonne de tickers depuis une table Wikipedia (repli)."""
    reponse = requests.get(url, headers=UA, timeout=30)
    reponse.raise_for_status()
    tables = pd.read_html(io.StringIO(reponse.text))
    for table in tables:
        if colonne in table.columns:
            valeurs = table[colonne].dropna().astype(str).tolist()
            return [v.strip().replace(".", "-").upper() for v in valeurs if v.strip()]
    return []


def univers_us():
    """
    Nasdaq 100 + S&P 500, dedoublonne.

    Source principale : yfiua/index-constituents (CSV statique, mis a jour
    mensuellement, tickers deja au format Yahoo Finance). Wikipedia sert de
    repli si ce service est indisponible : format HTML plus fragile, mais
    en cas de double echec on tombe sur tickers_us.txt en dernier recours.
    """
    tickers = []

    sources_csv = [
        ("https://yfiua.github.io/index-constituents/constituents-sp500.csv", "Symbol"),
        ("https://yfiua.github.io/index-constituents/constituents-nasdaq100.csv", "Symbol"),
    ]
    for url, colonne in sources_csv:
        try:
            trouves = _csv_colonne(url, colonne)
            if trouves:
                tickers.extend(trouves)
                print(f"  {len(trouves)} tickers depuis {url.split('/')[-1]}")
            else:
                print(f"  0 ticker depuis {url.split('/')[-1]} (colonne absente)")
        except Exception as erreur:
            print(f"  echec {url.split('/')[-1]}: {erreur}")

    if not tickers:
        print("  sources CSV indisponibles, repli sur Wikipedia")
        sources_wiki = [
            ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol"),
            ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "Ticker"),
            ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "Symbol"),
        ]
        for url, colonne in sources_wiki:
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


SLUGS_EURONEXT = {
    "paris": ("euronext-paris", ".PA"),
    "amsterdam": ("euronext-amsterdam", ".AS"),
    "bruxelles": ("euronext-brussels", ".BR"),
}


def _stockanalysis_page(slug, page):
    """Recupere une page de la liste stockanalysis.com (table complete)."""
    url = f"https://stockanalysis.com/list/{slug}/"
    if page > 1:
        url += f"?page={page}"
    reponse = requests.get(url, headers=UA, timeout=30)
    reponse.raise_for_status()
    tables = pd.read_html(io.StringIO(reponse.text))
    for table in tables:
        if "Symbol" in table.columns:
            return table
    return None


def univers_stockanalysis(slug, suffixe, max_pages=5):
    """
    Liste complete des valeurs cotees sur une place Euronext, via
    stockanalysis.com (triee par capitalisation, mise a jour quotidienne).
    Pagine automatiquement (500 lignes/page) jusqu'a la derniere page.
    Aucun filtre de capitalisation : tout ce qui est cote est inclus.
    """
    bruts = []
    for page in range(1, max_pages + 1):
        table = _stockanalysis_page(slug, page)
        if table is None or table.empty:
            break
        symboles = table["Symbol"].dropna().astype(str).tolist()
        if not symboles:
            break
        bruts.extend(symboles)
        if len(table) < 500:
            break

    tickers = []
    for symbole in bruts:
        symbole = symbole.strip().upper()
        if not symbole or symbole in ("-", "N/A"):
            continue
        tickers.append(f"{symbole}{suffixe}")

    return sorted(set(tickers))


def univers_indices():
    """
    Petite liste curee d'indices, pas de decouverte automatique (volume
    trop faible pour justifier une source dynamique).

    Depuis la migration EODHD, on utilise partout des indices CASH au
    format EODHD 'CODE.INDX' (ex: GSPC.INDX pour le S&P 500), plutot que
    des futures. Les futures (US comme ES=F, ou Euronext/Eurex FCE/FDAX)
    ne sont pas couverts par le palier EOD standard d'EODHD. L'indice cash
    suit le future de tres pres (arbitrage), donc reste un proxy fiable
    pour la detection Stromboli/Fernanda.

    Format 'CODE.INDX' confirme par EODHD pour GSPC.INDX (S&P 500) ; les
    autres suivent la meme convention documentee mais n'ont pas ete
    verifies individuellement — a confirmer via --valider-univers.
    """
    return [
        "GSPC.INDX", "NDX.INDX", "DJI.INDX", "RUT.INDX",   # US : S&P500, Nasdaq100, Dow, Russell2000
        "GDAXI.INDX", "FCHI.INDX", "STOXX50E.INDX",         # Europe : DAX, CAC40, Euro Stoxx 50
    ]


def univers_kraken_usd():
    """
    Toutes les paires cotees en USD (fiat, ZUSD) actives sur Kraken.
    Les paires USDT/USDC sont exclues pour eviter de tripler chaque crypto
    avec des paires quasi identiques.
    """
    try:
        reponse = requests.get(f"{KRAKEN_API}/AssetPairs", timeout=30)
        reponse.raise_for_status()
        data = reponse.json()
    except Exception as erreur:
        print(f"  echec AssetPairs Kraken : {erreur}")
        return []

    if data.get("error"):
        print(f"  erreur API Kraken : {data['error']}")
        return []

    paires = []
    for cle, info in data.get("result", {}).items():
        if info.get("quote") == "ZUSD" and info.get("status") == "online":
            paires.append(info.get("altname", cle))

    return sorted(set(paires))


def construire_univers(selection):
    """selection : 'tout', 'us', 'euronext', 'paris', 'amsterdam', 'bruxelles', 'indices', 'crypto'."""
    univers = {}

    if selection in ("tout", "us"):
        print("Univers US :")
        univers["US"] = univers_us()

    for place, (slug, suffixe) in SLUGS_EURONEXT.items():
        if selection in ("tout", "euronext", place):
            try:
                tickers = univers_stockanalysis(slug, suffixe)
            except Exception as erreur:
                print(f"  echec stockanalysis.com pour {place}: {erreur}")
                tickers = []

            if not tickers:
                fichier = FICHIERS_EURONEXT[place]
                print(f"  bascule sur le fichier de secours {fichier}")
                tickers = charger_fichier(RACINE / fichier)

            if tickers:
                print(f"Univers {place.capitalize()} : {len(tickers)} tickers")
                univers[place.capitalize()] = tickers

    if selection in ("tout", "indices"):
        tickers = univers_indices()
        print(f"Univers Indices : {len(tickers)} tickers")
        univers["Indices"] = tickers

    if selection in ("tout", "crypto"):
        tickers = univers_kraken_usd()
        if tickers:
            print(f"Univers Crypto (Kraken USD) : {len(tickers)} tickers")
            univers["Crypto"] = tickers
        else:
            print("  aucune paire crypto recuperee (Crypto absent de ce scan)")

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


def to_monthly(ohlc):
    """
    Agrege en bougies mensuelles (mois calendaire, cloture fin de mois).
    Le mois en cours, incomplet, est retire. Avec 2 ans d'historique, ca
    donne environ 24 bougies mensuelles : le seuil M7/Tenkan (9 periodes)
    est atteint, mais les Stromboli Monthly seront tres rares.

    Contrairement a to_weekly (dont le vendredi est presque toujours un
    jour de bourse), le dernier jour CALENDAIRE d'un mois tombe tres souvent
    un week-end ou un jour ferie. Comparer la derniere seance disponible a
    ce jour calendaire exact rejetterait alors a tort le mois qui vient de
    se terminer (ex : 31 mai un dimanche, derniere seance le vendredi 29 —
    le mois est bel et bien fini, mais 29 < 31). On considere donc le mois
    termine des que la derniere seance tombe dans les 3 derniers jours
    calendaires du mois (couvre les week-ends et la plupart des ponts feries).
    """
    mensuel = ohlc.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "Close"])

    if len(mensuel) == 0:
        return mensuel

    dernier_jour = ohlc.index[-1]
    proche_fin_de_mois = dernier_jour.day >= dernier_jour.days_in_month - 3
    if not proche_fin_de_mois:
        mensuel = mensuel.iloc[:-1]

    return mensuel


def agreger_tf(ohlc, tf):
    """Renvoie les bougies Daily/Weekly/Monthly selon tf ('D', 'W' ou 'M')."""
    if tf == "D":
        return ohlc
    if tf == "W":
        return to_weekly(ohlc)
    return to_monthly(ohlc)


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
    """
    Petit corps (<= SEUIL_DOJI % du range).

    Doji classique : meches des deux cotes.
    Doji dragonfly (si INCLURE_DRAGONFLY) : pas de meche haute, meche basse
    presente — signal de retournement haussier, parfois considere plus fort
    qu'un doji classique. Desactive par defaut pour ne pas changer le
    comportement existant du bot.
    """
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

    if meche_haute > EPS and meche_basse > EPS:
        return True  # doji classique

    if INCLURE_DRAGONFLY and meche_haute <= EPS and meche_basse > EPS:
        return True  # doji dragonfly

    return False


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
    Teste si la bougie d'indice i est le doji d'un Stromboli haussier.

    Seul le sens haussier est detecte pour le moment (le baissier/short
    n'est pas utilise cote trading, donc retire pour alleger calcul et
    alertes).
    """
    if i < MIN_BOUGIES:
        return None
    if not est_doji(ha, i):
        return None

    serie_rouge = compter_serie(ha, i - 1, est_rouge_pleine)
    if serie_rouge < MIN_BOUGIES:
        return None

    sens = "haussier"
    longueur = serie_rouge

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
# Fernanda
# ---------------------------------------------------------------------------
#
# Fernanda (long) : apres un Stromboli haussier, la bougie cloture au-dessus
#                    de la M7 (ascendante) et au-dessus de la Tenkan.
# Fernando/short retire pour le moment (non utilise cote trading).
#
# Le Stromboli sous-jacent reste "surveille" tant qu'aucune bougie ne fait
# un plus bas HA inferieur a celui de la bougie precedente. Des qu'une
# Fernanda se declenche, la surveillance de ce Stromboli s'arrete (pas de
# re-signal sur le meme setup).
#
# Comme 2 ans d'historique sont deja telecharges a chaque scan, tout se
# recalcule en une seule passe chronologique : aucun etat a stocker entre
# deux executions du bot.

def calcul_m7(ha):
    """Moyenne mobile simple 7 periodes sur la cloture HA."""
    return ha["close"].rolling(7).mean().to_numpy()


def calcul_tenkan(ha):
    """Tenkan-sen Ichimoku standard : (plus haut 9 + plus bas 9) / 2, sur HA."""
    haut9 = ha["high"].rolling(9).max()
    bas9 = ha["low"].rolling(9).min()
    return ((haut9 + bas9) / 2.0).to_numpy()


def detecter_fernanda_series(ha):
    """
    Parcourt toute la serie HA et retourne la liste chronologique des
    occurrences Fernanda, chacune liee au Stromboli haussier qui l'a
    declenchee.

    Seul le cote long (Fernanda) est detecte : Fernando/baissier est
    retire pour le moment, non utilise cote trading.
    """
    m7 = calcul_m7(ha)
    tenkan = calcul_tenkan(ha)
    ha_low = ha["low"].to_numpy()
    ha_close = ha["close"].to_numpy()

    occurrences = []
    actif_haussier = None

    for i in range(len(ha)):
        trouve = detecter_stromboli(ha, i)
        if trouve:
            actif_haussier = i

        if actif_haussier is not None and i > actif_haussier:
            valide = (
                ha_close[i] > m7[i]
                and m7[i] > m7[i - 1]
                and ha_close[i] > tenkan[i]
            )
            if valide:
                occurrences.append({
                    "type": "fernanda",
                    "index": i,
                    "date": ha.index[i],
                    "stromboli_date": ha.index[actif_haussier],
                    "stromboli_index": actif_haussier,
                })
                actif_haussier = None
            elif ha_low[i] < ha_low[i - 1]:
                actif_haussier = None

    return occurrences


def detecter_stromboli_baissier(ha, i):
    """
    Miroir de detecter_stromboli, cote baissier — RESERVE A L'ANALYSE
    (backtest --avec-fernando). N'est jamais appele par le scan live ni les
    alertes Telegram, qui restent long-only comme decide au depart.

    >= 3 bougies HA vertes PLEINES consecutives (aucune meche basse,
    HA_low == HA_open), suivies IMMEDIATEMENT d'un doji.
    """
    if i < MIN_BOUGIES:
        return None
    if not est_doji(ha, i):
        return None

    serie_verte = compter_serie(ha, i - 1, est_verte_pleine)
    if serie_verte < MIN_BOUGIES:
        return None

    etendue = _range(ha, i)
    corps = abs(float(ha["close"].iloc[i]) - float(ha["open"].iloc[i]))

    resultat = {
        "sens": "baissier",
        "bougies": serie_verte,
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


def detecter_fernando_series(ha):
    """
    Miroir de detecter_fernanda_series, cote baissier — RESERVE A L'ANALYSE.
    Fernando (short) : apres un Stromboli baissier, cloture sous la M7
    (descendante) et sous la Tenkan. Invalidation si une bougie fait un
    plus haut HA superieur a celui de la bougie precedente. Une seule
    Fernando par Stromboli baissier, meme regle de non re-signal que Fernanda.
    """
    m7 = calcul_m7(ha)
    tenkan = calcul_tenkan(ha)
    ha_high = ha["high"].to_numpy()
    ha_close = ha["close"].to_numpy()

    occurrences = []
    actif_baissier = None

    for i in range(len(ha)):
        trouve = detecter_stromboli_baissier(ha, i)
        if trouve:
            actif_baissier = i

        if actif_baissier is not None and i > actif_baissier:
            valide = (
                ha_close[i] < m7[i]
                and m7[i] < m7[i - 1]
                and ha_close[i] < tenkan[i]
            )
            if valide:
                occurrences.append({
                    "type": "fernando",
                    "index": i,
                    "date": ha.index[i],
                    "stromboli_date": ha.index[actif_baissier],
                    "stromboli_index": actif_baissier,
                })
                actif_baissier = None
            elif ha_high[i] > ha_high[i - 1]:
                actif_baissier = None

    return occurrences


# ---------------------------------------------------------------------------
# Backtest — probabilite de reussite des Fernanda
# ---------------------------------------------------------------------------
#
# Pour chaque Fernanda detectee dans l'historique, mesure le rendement REEL
# (prix de cloture reel, pas HA - c'est ce qu'on trade concretement) a
# plusieurs horizons apres l'entree. Le taux de reussite est la proportion
# de signaux dont le rendement est positif a cet horizon.

HORIZONS_BACKTEST = (1, 3, 5, 10, 20)


def ratio_volume(ha, i, fenetre=20):
    """
    Ratio volume de la bougie i / moyenne des `fenetre` bougies precedentes.
    None si le volume n'est pas disponible ou la moyenne est nulle.
    """
    if "volume" not in ha.columns:
        return None
    debut = max(0, i - fenetre)
    if i <= debut:
        return None
    moyenne = float(ha["volume"].iloc[debut:i].mean())
    if moyenne <= 0:
        return None
    return float(ha["volume"].iloc[i]) / moyenne


def backtest_fernanda(univers, annees, horizons=HORIZONS_BACKTEST, volume_min=None, avec_fernando=False):
    """
    Parcourt l'historique Daily de tout l'univers et releve le rendement reel
    (prix de cloture reel, pas HA) a plusieurs horizons, pour deux points
    d'entree possibles :
      - direct a la cloture du Stromboli (doji), sans attendre de confirmation
      - a la validation Fernanda (cloture au-dessus M7 ascendante + Tenkan)
    Ca permet de repondre a la question : attendre la Fernanda ameliore-t-il
    reellement les resultats, ou est-ce que trader des le Stromboli marche
    aussi bien (voire mieux, avec un point d'entree plus tot) ?

    volume_min : si fourni, ne garde que les signaux dont le Stromboli
    d'origine a un volume >= volume_min fois sa moyenne 20 bougies. C'est un
    outil d'ANALYSE uniquement (pour explorer si le volume au doji ameliore
    le taux de reussite) — le scan reel n'utilise jamais ce filtre, le volume
    y reste purement informatif, decision manuelle de l'operateur.

    avec_fernando : si True, calcule EN PLUS le miroir baissier (Stromboli
    baissier + Fernando/short), RESERVE A L'ANALYSE. Le "rendement" du short
    est l'inverse du rendement reel du prix (une baisse de prix = un gain
    pour le vendeur), pour que le taux de reussite se lise directement comme
    un vrai P&L de vente a decouvert, pas comme le simple miroir d'un
    rendement d'achat. Aucun impact sur le scan live ni les alertes
    Telegram, qui restent long-only.

    Retourne (DataFrame Stromboli, DataFrame Fernanda) si avec_fernando=False,
    ou (DataFrame Stromboli, DataFrame Fernanda, DataFrame Fernando) si True.
    """
    periode = f"{annees}y"
    lignes_stromboli = []
    lignes_fernanda = []
    lignes_fernando = []

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, cadre in donnees.items():
            if len(cadre) < MIN_BOUGIES + 30:
                continue

            ha = heikin_ashi(cadre)
            closes_reels = cadre["Close"].to_numpy(dtype=float)
            n = len(closes_reels)

            def rendements(i, short=False):
                prix_entree = closes_reels[i]
                ligne = {"prix_entree": prix_entree}
                for h in horizons:
                    j = i + h
                    if j < n and prix_entree > 0:
                        variation = (closes_reels[j] - prix_entree) / prix_entree * 100
                        ligne[f"rendement_{h}j"] = -variation if short else variation
                    else:
                        ligne[f"rendement_{h}j"] = None
                return ligne

            # Tous les Stromboli, qu'ils soient ensuite valides par une Fernanda ou pas
            for i in range(len(ha)):
                trouve = detecter_stromboli(ha, i)
                if trouve is None:
                    continue
                vol_ratio = ratio_volume(ha, i)
                if volume_min is not None and (vol_ratio is None or vol_ratio < volume_min):
                    continue
                lignes_stromboli.append({
                    "ticker": ticker, "place": place, "date": trouve["date"],
                    "volume_ratio_doji": vol_ratio,
                    **rendements(i),
                })

            # Fernanda (entree confirmee) — le filtre volume porte sur le doji
            # du Stromboli d'origine, pas sur la bougie de validation elle-meme :
            # c'est la participation au moment du retournement qui nous interesse.
            for occ in detecter_fernanda_series(ha):
                i = occ["index"]
                vol_ratio = ratio_volume(ha, occ["stromboli_index"])
                if volume_min is not None and (vol_ratio is None or vol_ratio < volume_min):
                    continue
                lignes_fernanda.append({
                    "ticker": ticker, "place": place, "date": occ["date"],
                    "stromboli_date": occ["stromboli_date"],
                    "volume_ratio_doji": vol_ratio,
                    **rendements(i),
                })

            # Fernando (short, analyse uniquement)
            if avec_fernando:
                for occ in detecter_fernando_series(ha):
                    i = occ["index"]
                    vol_ratio = ratio_volume(ha, occ["stromboli_index"])
                    if volume_min is not None and (vol_ratio is None or vol_ratio < volume_min):
                        continue
                    lignes_fernando.append({
                        "ticker": ticker, "place": place, "date": occ["date"],
                        "stromboli_date": occ["stromboli_date"],
                        "volume_ratio_doji": vol_ratio,
                        **rendements(i, short=True),
                    })

    if avec_fernando:
        return pd.DataFrame(lignes_stromboli), pd.DataFrame(lignes_fernanda), pd.DataFrame(lignes_fernando)
    return pd.DataFrame(lignes_stromboli), pd.DataFrame(lignes_fernanda)


def _table_horizons(df, horizons):
    """Lignes formatees reussite/rendement par horizon, pour un DataFrame de signaux."""
    lignes = []
    for h in horizons:
        col = f"rendement_{h}j"
        valides = df[col].dropna()
        if len(valides) == 0:
            continue
        taux_reussite = (valides > 0).mean() * 100
        lignes.append(
            f"  {h:>2}j : {len(valides):>4} signaux exploitables · "
            f"reussite {taux_reussite:5.1f}% · "
            f"rendement moyen {valides.mean():+6.2f}% · "
            f"median {valides.median():+6.2f}%"
        )
    return lignes


def resume_backtest(df_stromboli, df_fernanda, annees, volume_min=None, horizons=HORIZONS_BACKTEST, df_fernando=None):
    total_stromboli = len(df_stromboli)
    total_fernanda = len(df_fernanda)
    taux_validation = (total_fernanda / total_stromboli * 100) if total_stromboli else 0.0

    entete = f"Backtest — {annees} ans"
    if volume_min is not None:
        entete += f" · filtre volume >= x{volume_min} au doji (analyse uniquement, jamais applique en scan reel)"

    sortie = [
        entete,
        f"  Stromboli detectes : {total_stromboli} · "
        f"Fernanda : {total_fernanda} · "
        f"taux de validation {taux_validation:.1f}%",
        "",
    ]

    sortie.append(f"ENTREE DIRECTE AU STROMBOLI ({total_stromboli} signaux)")
    if df_stromboli.empty:
        sortie.append("  aucun signal exploitable")
    else:
        sortie.extend(_table_horizons(df_stromboli, horizons))
    sortie.append("")

    sortie.append(f"ENTREE A LA FERNANDA ({total_fernanda} signaux)")
    if df_fernanda.empty:
        sortie.append("  aucun signal exploitable")
    else:
        sortie.extend(_table_horizons(df_fernanda, horizons))
    sortie.append("")

    if df_fernando is not None:
        total_fernando = len(df_fernando)
        sortie.append(
            f"ENTREE AU FERNANDO / SHORT ({total_fernando} signaux) "
            f"— analyse uniquement, jamais en scan reel"
        )
        sortie.append(
            "  Rendement = P&L reel d'une vente a decouvert "
            "(une baisse de prix apparait en positif)"
        )
        if df_fernando.empty:
            sortie.append("  aucun signal exploitable")
        else:
            sortie.extend(_table_horizons(df_fernando, horizons))
        sortie.append("")

    if not df_fernanda.empty:
        sortie.append("Fernanda par place :")
        sortie.append(str(df_fernanda.groupby("place").size().rename("signaux")))

    if df_fernando is not None and not df_fernando.empty:
        sortie.append("")
        sortie.append("Fernando par place :")
        sortie.append(str(df_fernando.groupby("place").size().rename("signaux")))

    return "\n".join(sortie)


# ---------------------------------------------------------------------------
# Backtest — retour a la moyenne (RSI-2 / IBS + filtre MM200)
# ---------------------------------------------------------------------------
#
# Setup independant de la methode Inchi, sur prix REELS (pas Heikin Ashi) :
#   Filtre     : cloture > MM200 et MM200 ascendante (repli dans une tendance
#                de fond haussiere, jamais a contre-tendance)
#   Declencheur: RSI-2 < 10  (variante 'rsi2')  ou  IBS < 0.20 (variante 'ibs')
#   Entree     : cloture du jour du signal
#   Sortie     : premiere cloture > MM5, ou RSI-2 > 70, ou MAX_HOLD seances
#   Un seul trade a la fois par ticker (pas de chevauchement).
#
# Contrairement au backtest Inchi (horizons fixes), on mesure ici un vrai
# P&L par trade avec la regle de sortie du setup : c'est la seule facon de
# comparer honnetement au ~70% publie par Connors pour le RSI-2.
# Outil d'ANALYSE uniquement : rien de tout ca n'est branche sur le scan live.

RM_SEUIL_RSI2 = 10.0
RM_SEUIL_IBS = 0.20
RM_RSI2_SORTIE = 70.0
RM_MAX_HOLD = 10


def calcul_rsi(closes, periode=2):
    """RSI de Wilder (lissage exponentiel classique). Retourne un ndarray (NaN au debut)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    rsi = np.full(n, np.nan)
    if n <= periode:
        return rsi
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    pertes = np.where(delta < 0, -delta, 0.0)
    gain_moy = gains[:periode].mean()
    perte_moy = pertes[:periode].mean()
    for i in range(periode, n - 1):
        if i > periode:
            gain_moy = (gain_moy * (periode - 1) + gains[i - 1]) / periode
            perte_moy = (perte_moy * (periode - 1) + pertes[i - 1]) / periode
        if perte_moy == 0:
            rsi[i] = 100.0 if gain_moy > 0 else 50.0
        else:
            rs = gain_moy / perte_moy
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    # derniere valeur
    gain_moy = (gain_moy * (periode - 1) + gains[-1]) / periode
    perte_moy = (perte_moy * (periode - 1) + pertes[-1]) / periode
    rsi[-1] = 100.0 if perte_moy == 0 and gain_moy > 0 else (
        50.0 if perte_moy == 0 else 100.0 - 100.0 / (1.0 + gain_moy / perte_moy)
    )
    return rsi


def calcul_ibs(cadre):
    """IBS = (Close - Low) / (High - Low). NaN si range nul."""
    haut = cadre["High"].to_numpy(dtype=float)
    bas = cadre["Low"].to_numpy(dtype=float)
    cloture = cadre["Close"].to_numpy(dtype=float)
    etendue = haut - bas
    with np.errstate(divide="ignore", invalid="ignore"):
        ibs = np.where(etendue > 0, (cloture - bas) / etendue, np.nan)
    return ibs


def _donnee_suspecte(cadre, gap_prix_max=80.0, gap_jours_max=60):
    """
    Detecte les series de prix qui portent la signature d'une suspension de
    cotation prolongee suivie d'une reprise chaotique (ex: ALTRA.PA, trou de
    cotation 2020-2025 puis +198% en une seule seance a la reprise). Deux
    signaux independants, l'un OU l'autre suffit :
      - un saut de prix extreme d'une seance a l'autre (gap_prix_max, en %)
      - un trou de plusieurs mois dans le calendrier de cotation lui-meme
        (gap_jours_max, en jours calendaires) — le signal le plus specifique :
        un vrai mouvement de marche, meme violent, ne cree jamais un trou de
        plusieurs mois dans les dates, seule une suspension le fait.

    Le ticker entier est exclu (pas seulement le trade concerne) car toute
    la fenetre MM200/EMA autour de l'evenement est faussee par des prix
    d'avant-suspension totalement decorreles du niveau de reprise.
    """
    closes = cadre["Close"].to_numpy(dtype=float)
    if len(closes) < 2:
        return False

    variations = np.abs(np.diff(closes) / closes[:-1] * 100)
    if len(variations) and np.nanmax(variations) > gap_prix_max:
        return True

    ecarts_jours = cadre.index.to_series().diff().dt.days.to_numpy()[1:]
    if len(ecarts_jours) and np.nanmax(ecarts_jours) > gap_jours_max:
        return True

    return False


def _ratio_volume_signal(cadre, i, fenetre=20):
    """
    Ratio volume du jour i / moyenne des `fenetre` jours precedents, sur un
    DataFrame OHLCV brut (colonne 'Volume'). Teste l'hypothese 'pic de
    volume au moment du signal' : un fort volume ce jour-la est-il un bon
    ou un mauvais signe pour le retour a la moyenne ?
    """
    if "Volume" not in cadre.columns:
        return None
    debut = max(0, i - fenetre)
    if i <= debut:
        return None
    moyenne = float(cadre["Volume"].iloc[debut:i].mean())
    if moyenne <= 0:
        return None
    return float(cadre["Volume"].iloc[i]) / moyenne


def _liquidite_moyenne(cadre, i, fenetre=60):
    """
    Volume moyen (nombre de titres/jour) sur les `fenetre` jours precedant
    le signal — mesure de liquidite GENERALE du titre, independante du
    signal lui-meme. Teste l'hypothese 'les titres liquides performent-ils
    mieux que les micro-caps peu tradees ?', separement du pic ponctuel.
    """
    if "Volume" not in cadre.columns:
        return None
    debut = max(0, i - fenetre)
    if i <= debut:
        return None
    return float(cadre["Volume"].iloc[debut:i].mean())


def _trades_retour_moyenne(
    cadre, declencheur, stop_pct=None, frais_pct=0.0,
    volume_min_signal=None, volume_min_liquidite=None, entree_lendemain=False,
):
    """
    Simule les trades d'un ticker pour un declencheur donne ('rsi2' ou 'ibs').
    stop_pct : si fourni (ex: -8.0 pour -8%), sortie immediate au niveau du
    stop des que le plus bas de la seance le franchit (avant meme de tester
    les conditions de sortie normales ce jour-la — un stop protege contre
    la baisse intra-seance, pas seulement a la cloture).
    frais_pct : frais de courtage aller-retour, en % de la taille de position
    (ex: 0.2 pour 2€ de frais sur une position de 1000€), deduits directement
    du rendement de chaque trade. 0.0 par defaut = aucun frais (comportement
    inchange). A calibrer selon TON compte reel — jamais suppose par le code.
    volume_min_signal : si fourni, ne garde que les signaux dont le volume
    DU JOUR est >= volume_min_signal fois sa moyenne 20 jours (pic de volume
    au signal). Outil d'ANALYSE uniquement, jamais applique en scan reel.
    volume_min_liquidite : si fourni, ne garde que les signaux dont le
    volume MOYEN des 60 jours precedents (hors du signal lui-meme, mesure
    de liquidite generale du titre) est >= ce seuil (nombre absolu de
    titres/jour). Outil d'ANALYSE uniquement, jamais applique en scan reel.
    entree_lendemain : si True, simule une entree REALISTE a l'OUVERTURE du
    jour SUIVANT le signal (le bot ne peut agir qu'apres cloture, donc le
    premier ordre reellement possible est le lendemain matin), plutot qu'a
    la cloture du jour du signal lui-meme — hypothese optimiste et
    impossible a executer en pratique, utilisee par defaut dans toutes les
    versions precedentes du backtest. Permet de mesurer l'ecart reel entre
    theorie et execution.
    Retourne une liste de dicts (date_entree, date_sortie, jours, rendement, motif).
    """
    closes = cadre["Close"].to_numpy(dtype=float)
    ouvertures = cadre["Open"].to_numpy(dtype=float)
    bas = cadre["Low"].to_numpy(dtype=float)
    n = len(closes)
    if n < 210:
        return []

    mm200 = pd.Series(closes).rolling(200).mean().to_numpy()
    mm5 = pd.Series(closes).rolling(5).mean().to_numpy()
    rsi2 = calcul_rsi(closes, 2)
    ibs = calcul_ibs(cadre) if declencheur == "ibs" else None

    trades = []
    i = 200
    while i < n - 1:
        filtre_ok = (
            not np.isnan(mm200[i]) and not np.isnan(mm200[i - 1])
            and closes[i] > mm200[i] and mm200[i] > mm200[i - 1]
        )
        if declencheur == "rsi2":
            signal = filtre_ok and not np.isnan(rsi2[i]) and rsi2[i] < RM_SEUIL_RSI2
        else:
            signal = filtre_ok and not np.isnan(ibs[i]) and ibs[i] < RM_SEUIL_IBS

        if not signal:
            i += 1
            continue

        if volume_min_signal is not None:
            ratio = _ratio_volume_signal(cadre, i)
            if ratio is None or ratio < volume_min_signal:
                i += 1
                continue

        if volume_min_liquidite is not None:
            liquidite = _liquidite_moyenne(cadre, i)
            if liquidite is None or liquidite < volume_min_liquidite:
                i += 1
                continue

        if entree_lendemain:
            entree_idx = i + 1  # toujours < n car la boucle exige i < n-1
            prix_entree = ouvertures[entree_idx]
            debut_recherche = entree_idx  # le reste de la seance d'entree compte deja
        else:
            entree_idx = i
            prix_entree = closes[i]
            debut_recherche = entree_idx + 1

        prix_stop = prix_entree * (1 + stop_pct / 100) if stop_pct is not None else None
        sortie_j, motif, prix_sortie = None, None, None

        for j in range(debut_recherche, min(entree_idx + RM_MAX_HOLD, n - 1) + 1):
            if prix_stop is not None and bas[j] <= prix_stop:
                sortie_j, motif, prix_sortie = j, "stop_loss", prix_stop
                break
            if not np.isnan(mm5[j]) and closes[j] > mm5[j]:
                sortie_j, motif, prix_sortie = j, "mm5", closes[j]
                break
            if not np.isnan(rsi2[j]) and rsi2[j] > RM_RSI2_SORTIE:
                sortie_j, motif, prix_sortie = j, "rsi70", closes[j]
                break

        if sortie_j is None:
            sortie_j = min(entree_idx + RM_MAX_HOLD, n - 1)
            motif, prix_sortie = "max_hold", closes[sortie_j]

        rendement_brut = (prix_sortie - prix_entree) / prix_entree * 100
        trades.append({
            "date_entree": cadre.index[entree_idx],
            "date_sortie": cadre.index[sortie_j],
            "jours": sortie_j - entree_idx,
            "rendement": rendement_brut - frais_pct,
            "motif": motif,
        })
        i = sortie_j + 1  # pas de chevauchement

    return trades


def backtest_retour_moyenne(
    univers, annees, declencheurs=("rsi2", "ibs"), stop_pct=None, frais_pct=0.0,
    volume_min_signal=None, volume_min_liquidite=None, entree_lendemain=False,
):
    """Lance les simulations sur tout l'univers. Retourne {declencheur: DataFrame}."""
    periode = f"{annees}y"
    resultats = {d: [] for d in declencheurs}
    exclus = 0

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, cadre in donnees.items():
            if _donnee_suspecte(cadre):
                exclus += 1
                continue
            for d in declencheurs:
                for t in _trades_retour_moyenne(
                    cadre, d, stop_pct=stop_pct, frais_pct=frais_pct,
                    volume_min_signal=volume_min_signal, volume_min_liquidite=volume_min_liquidite,
                    entree_lendemain=entree_lendemain,
                ):
                    resultats[d].append({"ticker": ticker, "place": place, **t})

    if exclus:
        print(f"\n{exclus} tickers exclus (saut de prix ou trou de cotation suspect)")

    return {d: pd.DataFrame(lignes) for d, lignes in resultats.items()}


def _stats_trades(df):
    """Lignes formatees (reussite/rendement/duree/gagnants-perdants/sorties) pour un DataFrame de trades."""
    if df.empty:
        return ["  aucun trade"]
    r = df["rendement"]
    gagnants = r[r > 0]
    perdants = r[r <= 0]
    lignes = [
        f"  reussite {(r > 0).mean() * 100:5.1f}% · "
        f"rendement moyen {r.mean():+6.2f}% · median {r.median():+6.2f}% · "
        f"duree moyenne {df['jours'].mean():.1f} seances ({len(df)} trades)",
        f"  gain moyen des gagnants {gagnants.mean() if len(gagnants) else 0:+6.2f}% · "
        f"perte moyenne des perdants {perdants.mean() if len(perdants) else 0:+6.2f}% · "
        f"pire trade {r.min():+6.2f}%",
    ]
    motifs = df["motif"].value_counts(normalize=True) * 100
    lignes.append("  sorties : " + " · ".join(f"{m} {p:.0f}%" for m, p in motifs.items()))
    if "place" in df.columns:
        lignes.append("  par place : " + " · ".join(
            f"{p} {len(g)} trades / {(g['rendement'] > 0).mean() * 100:.0f}%"
            for p, g in df.groupby("place")
        ))
    return lignes


def resume_retour_moyenne(
    resultats, annees, stop_pct=None, frais_pct=0.0,
    volume_min_signal=None, volume_min_liquidite=None, entree_lendemain=False,
):
    libelles = {"rsi2": f"RSI-2 < {RM_SEUIL_RSI2:.0f}", "ibs": f"IBS < {RM_SEUIL_IBS:.2f}"}
    entete = f"Backtest retour a la moyenne — {annees} ans (analyse uniquement, jamais en scan reel)"
    sortie = [
        entete,
        f"  Filtre : cloture > MM200 ascendante · Sortie : cloture > MM5 ou RSI-2 > "
        f"{RM_RSI2_SORTIE:.0f} ou {RM_MAX_HOLD} seances max"
        + (f" · Stop loss : {stop_pct:+.1f}%" if stop_pct is not None else "")
        + (f" · Frais : -{frais_pct:.2f}% par trade (calibrer selon TON compte reel)" if frais_pct else "")
        + (f" · Pic volume >= x{volume_min_signal:.1f} au signal" if volume_min_signal is not None else "")
        + (f" · Liquidite moyenne >= {volume_min_liquidite:,.0f} titres/jour" if volume_min_liquidite is not None else "")
        + (" · Entree REALISTE a l'ouverture du lendemain (pas la cloture du signal)" if entree_lendemain else ""),
        "",
    ]
    for d, df in resultats.items():
        sortie.append(f"DECLENCHEUR {libelles.get(d, d)} ({len(df)} trades)")
        sortie.extend(_stats_trades(df))
        sortie.append("")
    return "\n".join(sortie)


def resume_validation_croisee(resultats, annees, stop_pct=None):
    """
    Decoupe chaque declencheur en deux moities CHRONOLOGIQUES par date
    d'entree (coupure = date mediane des trades) : 'decouverte' (premiere
    moitie) et 'validation' (seconde moitie, hors echantillon). Un edge
    reel doit tenir sur les deux ; s'il ne fonctionne que sur la decouverte,
    c'est un mirage statistique (surapprentissage sur la periode testee).
    """
    libelles = {"rsi2": f"RSI-2 < {RM_SEUIL_RSI2:.0f}", "ibs": f"IBS < {RM_SEUIL_IBS:.2f}"}
    sortie = [
        f"Validation croisee (decouverte / validation) — {annees} ans"
        + (f" · Stop loss : {stop_pct:+.1f}%" if stop_pct is not None else ""),
        "  Coupure = date mediane des trades. Un edge reel doit tenir sur les DEUX moities.",
        "",
    ]
    for d, df in resultats.items():
        sortie.append(f"DECLENCHEUR {libelles.get(d, d)}")
        if df.empty or len(df) < 20:
            sortie.append("  echantillon trop petit pour decouper (< 20 trades)")
            sortie.append("")
            continue

        coupure = df["date_entree"].median()
        decouverte = df[df["date_entree"] < coupure]
        validation = df[df["date_entree"] >= coupure]

        sortie.append(f"  DECOUVERTE (avant {coupure.date()})")
        sortie.extend("  " + l for l in _stats_trades(decouverte))
        sortie.append(f"  VALIDATION (a partir de {coupure.date()}, hors echantillon)")
        sortie.extend("  " + l for l in _stats_trades(validation))
        sortie.append("")

    return "\n".join(sortie)


# ---------------------------------------------------------------------------
# Backtest — suivi de tendance (EMA 8/21, filtre EMA50, Chandelier Exit)
# ---------------------------------------------------------------------------
#
# Setup independant de la methode Inchi, sur prix REELS, famille CONTINUATION
# de tendance (pas retour a la moyenne comme RSI-2/IBS) :
#   Filtre     : EMA21 > EMA50 (tendance de fond confirmee)
#   Entree     : EMA8 croise AU-DESSUS de l'EMA21 (le jour du croisement)
#   Sortie     : stop suiveur Chandelier Exit (plus haut depuis l'entree
#                moins 3x ATR22), OU l'EMA8 recroise sous l'EMA21 —
#                AUCUNE limite de duree fixe : on laisse courir la tendance
#                tant qu'elle ne montre pas de vrai signe de faiblesse.
# Profil attendu, oppose au retour a la moyenne : moins de trades gagnants,
# mais des gagnants nettement plus gros (on laisse courir la tendance).
# Un seul trade a la fois par ticker. Outil d'ANALYSE uniquement.

RM_EMA_RAPIDE = 8
RM_EMA_LENTE = 21
RM_EMA_FOND = 50
RM_CHANDELIER_ATR_PERIODE = 22
RM_CHANDELIER_MULTIPLICATEUR = 3.0


def calcul_ema(closes, periode):
    """EMA standard (pandas ewm, adjust=False)."""
    return pd.Series(closes, dtype=float).ewm(span=periode, adjust=False).mean().to_numpy()


def calcul_atr(cadre, periode=RM_CHANDELIER_ATR_PERIODE):
    """
    ATR de Wilder (True Range lisse exponentiellement, methode standard).
    True Range = max(High-Low, |High-Close_veille|, |Low-Close_veille|).
    """
    haut = cadre["High"].to_numpy(dtype=float)
    bas = cadre["Low"].to_numpy(dtype=float)
    cloture = cadre["Close"].to_numpy(dtype=float)
    n = len(cloture)

    tr = np.empty(n)
    tr[0] = haut[0] - bas[0]
    for i in range(1, n):
        tr[i] = max(
            haut[i] - bas[i],
            abs(haut[i] - cloture[i - 1]),
            abs(bas[i] - cloture[i - 1]),
        )

    atr = np.full(n, np.nan)
    if n <= periode:
        return atr
    atr[periode] = tr[1:periode + 1].mean()
    for i in range(periode + 1, n):
        atr[i] = (atr[i - 1] * (periode - 1) + tr[i]) / periode
    return atr


def _trades_ema_cross(cadre, stop_pct=None, frais_pct=0.0):
    """
    Simule les trades de suivi de tendance EMA8/21 pour un ticker.
    Sortie principale : Chandelier Exit (stop suiveur base sur l'ATR du
    titre, pas un pourcentage fixe identique pour tous). stop_pct, si
    fourni, agit comme plancher de securite supplementaire (perte max
    absolue depuis l'entree), en plus du Chandelier — jamais a la place.
    Aucune limite de duree : le trade court tant qu'aucune des deux
    conditions de sortie n'est declenchee.
    """
    closes = cadre["Close"].to_numpy(dtype=float)
    haut = cadre["High"].to_numpy(dtype=float)
    bas = cadre["Low"].to_numpy(dtype=float)
    n = len(closes)
    if n < RM_EMA_FOND + RM_CHANDELIER_ATR_PERIODE + 10:
        return []

    ema_rapide = calcul_ema(closes, RM_EMA_RAPIDE)
    ema_lente = calcul_ema(closes, RM_EMA_LENTE)
    ema_fond = calcul_ema(closes, RM_EMA_FOND)
    atr = calcul_atr(cadre)

    trades = []
    i = max(RM_EMA_FOND, RM_CHANDELIER_ATR_PERIODE)
    while i < n - 1:
        croisement_haussier = (
            ema_rapide[i] > ema_lente[i] and ema_rapide[i - 1] <= ema_lente[i - 1]
        )
        filtre_ok = ema_lente[i] > ema_fond[i]
        if not (croisement_haussier and filtre_ok) or np.isnan(atr[i]):
            i += 1
            continue

        prix_entree = closes[i]
        prix_stop_fixe = prix_entree * (1 + stop_pct / 100) if stop_pct is not None else None
        plus_haut = haut[i]
        sortie_j, motif, prix_sortie = None, None, None

        j = i + 1
        while j < n:
            niveau_chandelier = plus_haut - RM_CHANDELIER_MULTIPLICATEUR * atr[j]

            if prix_stop_fixe is not None and bas[j] <= prix_stop_fixe:
                sortie_j, motif, prix_sortie = j, "stop_fixe", prix_stop_fixe
                break
            if bas[j] <= niveau_chandelier:
                sortie_j, motif, prix_sortie = j, "chandelier", niveau_chandelier
                break
            if ema_rapide[j] < ema_lente[j]:
                sortie_j, motif, prix_sortie = j, "croisement_baissier", closes[j]
                break

            plus_haut = max(plus_haut, haut[j])
            j += 1

        if sortie_j is None:
            sortie_j = n - 1
            motif, prix_sortie = "fin_donnees", closes[sortie_j]

        rendement_brut = (prix_sortie - prix_entree) / prix_entree * 100
        trades.append({
            "date_entree": cadre.index[i],
            "date_sortie": cadre.index[sortie_j],
            "jours": sortie_j - i,
            "rendement": rendement_brut - frais_pct,
            "motif": motif,
        })
        i = sortie_j + 1  # pas de chevauchement

    return trades


def backtest_ema_cross(univers, annees, stop_pct=None, frais_pct=0.0):
    """Lance le backtest EMA 8/21 sur tout l'univers. Retourne {'ema_cross': DataFrame}."""
    periode = f"{annees}y"
    lignes = []
    exclus = 0

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, cadre in donnees.items():
            if _donnee_suspecte(cadre):
                exclus += 1
                continue
            for t in _trades_ema_cross(cadre, stop_pct=stop_pct, frais_pct=frais_pct):
                lignes.append({"ticker": ticker, "place": place, **t})

    if exclus:
        print(f"\n{exclus} tickers exclus (saut de prix ou trou de cotation suspect)")

    return {"ema_cross": pd.DataFrame(lignes)}


def resume_ema_cross(resultats, annees, stop_pct=None, frais_pct=0.0):
    entete = f"Backtest suivi de tendance EMA {RM_EMA_RAPIDE}/{RM_EMA_LENTE} — {annees} ans (analyse uniquement, jamais en scan reel)"
    sortie = [
        entete,
        f"  Filtre : EMA{RM_EMA_LENTE} > EMA{RM_EMA_FOND} · "
        f"Entree : EMA{RM_EMA_RAPIDE} croise au-dessus EMA{RM_EMA_LENTE} · "
        f"Sortie : Chandelier Exit (plus haut - {RM_CHANDELIER_MULTIPLICATEUR:.0f}xATR{RM_CHANDELIER_ATR_PERIODE}) "
        f"ou croisement inverse — aucune limite de duree"
        + (f" · Plancher de securite : {stop_pct:+.1f}%" if stop_pct is not None else "")
        + (f" · Frais : -{frais_pct:.2f}% par trade" if frais_pct else ""),
        "",
    ]
    for d, df in resultats.items():
        sortie.append(f"DECLENCHEUR {d} ({len(df)} trades)")
        sortie.extend(_stats_trades(df))
        sortie.append("")
    return "\n".join(sortie)


# ---------------------------------------------------------------------------
# Backtest — suivi de tendance avec SORTIE PARTIELLE (compromis reussite/gain)
# ---------------------------------------------------------------------------
#
# Meme entree que ema-cross (croisement EMA8/21, filtre EMA50). La position
# est scindee en deux moities des l'entree :
#   Moitie A : sort au premier des evenements suivants : objectif fixe
#              atteint (CIBLE_PARTIELLE, ex +5%), Chandelier Exit, ou
#              croisement EMA inverse — vise un taux de reussite plus eleve
#              en encaissant un petit gain rapide.
#   Moitie B : reste en Chandelier Exit pur (comme ema-cross), pour capter
#              les gros mouvements quand ils arrivent.
# Le rendement rapporte est la moyenne 50/50 des deux moities. Un vrai
# compromis mesurable, pas un cumul des deux avantages sans contrepartie.
# Outil d'ANALYSE uniquement.

RM_SORTIE_PARTIELLE_CIBLE = 5.0


def _trades_ema_cross_partiel(cadre, cible_partielle=RM_SORTIE_PARTIELLE_CIBLE, stop_pct=None, frais_pct=0.0):
    """Simule les trades de la variante 'sortie partielle'. Meme structure que _trades_ema_cross."""
    closes = cadre["Close"].to_numpy(dtype=float)
    haut = cadre["High"].to_numpy(dtype=float)
    bas = cadre["Low"].to_numpy(dtype=float)
    n = len(closes)
    if n < RM_EMA_FOND + RM_CHANDELIER_ATR_PERIODE + 10:
        return []

    ema_rapide = calcul_ema(closes, RM_EMA_RAPIDE)
    ema_lente = calcul_ema(closes, RM_EMA_LENTE)
    ema_fond = calcul_ema(closes, RM_EMA_FOND)
    atr = calcul_atr(cadre)

    trades = []
    i = max(RM_EMA_FOND, RM_CHANDELIER_ATR_PERIODE)
    while i < n - 1:
        croisement_haussier = (
            ema_rapide[i] > ema_lente[i] and ema_rapide[i - 1] <= ema_lente[i - 1]
        )
        filtre_ok = ema_lente[i] > ema_fond[i]
        if not (croisement_haussier and filtre_ok) or np.isnan(atr[i]):
            i += 1
            continue

        prix_entree = closes[i]
        prix_cible = prix_entree * (1 + cible_partielle / 100)
        prix_stop_fixe = prix_entree * (1 + stop_pct / 100) if stop_pct is not None else None
        plus_haut = haut[i]

        moitie_a, moitie_b = None, None
        j = i + 1
        while j < n and (moitie_a is None or moitie_b is None):
            niveau_chandelier = plus_haut - RM_CHANDELIER_MULTIPLICATEUR * atr[j]

            if moitie_a is None:
                if prix_stop_fixe is not None and bas[j] <= prix_stop_fixe:
                    moitie_a = (j, "stop_fixe", prix_stop_fixe)
                elif haut[j] >= prix_cible:
                    moitie_a = (j, "cible_partielle", prix_cible)
                elif bas[j] <= niveau_chandelier:
                    moitie_a = (j, "chandelier", niveau_chandelier)
                elif ema_rapide[j] < ema_lente[j]:
                    moitie_a = (j, "croisement_baissier", closes[j])

            if moitie_b is None:
                if prix_stop_fixe is not None and bas[j] <= prix_stop_fixe:
                    moitie_b = (j, "stop_fixe", prix_stop_fixe)
                elif bas[j] <= niveau_chandelier:
                    moitie_b = (j, "chandelier", niveau_chandelier)
                elif ema_rapide[j] < ema_lente[j]:
                    moitie_b = (j, "croisement_baissier", closes[j])

            plus_haut = max(plus_haut, haut[j])
            j += 1

        if moitie_a is None:
            moitie_a = (n - 1, "fin_donnees", closes[n - 1])
        if moitie_b is None:
            moitie_b = (n - 1, "fin_donnees", closes[n - 1])

        rendement_a = (moitie_a[2] - prix_entree) / prix_entree * 100
        rendement_b = (moitie_b[2] - prix_entree) / prix_entree * 100
        rendement_blend = 0.5 * rendement_a + 0.5 * rendement_b - frais_pct

        sortie_finale_j = max(moitie_a[0], moitie_b[0])
        trades.append({
            "date_entree": cadre.index[i],
            "date_sortie": cadre.index[sortie_finale_j],
            "jours": sortie_finale_j - i,
            "rendement": rendement_blend,
            "motif": f"{moitie_a[1]}+{moitie_b[1]}",
        })
        i = sortie_finale_j + 1

    return trades


def backtest_ema_cross_partiel(univers, annees, cible_partielle=RM_SORTIE_PARTIELLE_CIBLE, stop_pct=None, frais_pct=0.0):
    """Lance le backtest 'sortie partielle' sur tout l'univers. Retourne {'ema_cross_partiel': DataFrame}."""
    periode = f"{annees}y"
    lignes = []
    exclus = 0

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, cadre in donnees.items():
            if _donnee_suspecte(cadre):
                exclus += 1
                continue
            for t in _trades_ema_cross_partiel(cadre, cible_partielle=cible_partielle, stop_pct=stop_pct, frais_pct=frais_pct):
                lignes.append({"ticker": ticker, "place": place, **t})

    if exclus:
        print(f"\n{exclus} tickers exclus (saut de prix ou trou de cotation suspect)")

    return {"ema_cross_partiel": pd.DataFrame(lignes)}


def resume_ema_cross_partiel(resultats, annees, cible_partielle=RM_SORTIE_PARTIELLE_CIBLE, stop_pct=None, frais_pct=0.0):
    entete = f"Backtest suivi de tendance — sortie partielle — {annees} ans (analyse uniquement, jamais en scan reel)"
    sortie = [
        entete,
        f"  Meme entree que ema-cross. Moitie A : sort a +{cible_partielle:.1f}% (ou Chandelier/croisement "
        f"si atteint avant). Moitie B : Chandelier Exit pur, aucune limite de duree. "
        f"Rendement = moyenne 50/50 des deux moities."
        + (f" · Plancher de securite : {stop_pct:+.1f}%" if stop_pct is not None else "")
        + (f" · Frais : -{frais_pct:.2f}% par trade" if frais_pct else ""),
        "",
    ]
    for d, df in resultats.items():
        sortie.append(f"DECLENCHEUR {d} ({len(df)} trades)")
        sortie.extend(_stats_trades(df))
        sortie.append("")
    return "\n".join(sortie)


def diagnostiquer(ticker, tf, nb_bougies=25):
    """
    Affiche, bougie par bougie, les valeurs HA exactes calculees par le bot
    pour un ticker donne, avec le statut de chaque bougie (rouge pleine,
    doji, stromboli detecte, invalidation). Sert a comparer chiffre par
    chiffre avec un autre graphique (TradingView etc.) plutot qu'a l'oeil.
    """
    print(f"Telechargement de {ticker}...")
    donnees = telecharger([ticker], PERIODE_DAILY)
    if ticker not in donnees:
        print(f"Aucune donnee recuperee pour {ticker}.")
        return

    cadre = agreger_tf(donnees[ticker], tf)
    if len(cadre) < MIN_BOUGIES + 2:
        print("Pas assez de bougies.")
        return

    ha = heikin_ashi(cadre)
    m7 = calcul_m7(ha)
    tenkan = calcul_tenkan(ha)

    actif_haussier = None
    statuts = {}

    for i in range(len(ha)):
        trouve = detecter_stromboli(ha, i)
        if trouve:
            actif_haussier = i
            statuts[i] = "STROMBOLI (doji)"

        if actif_haussier is not None and i > actif_haussier:
            valide = (
                ha["close"].iloc[i] > m7[i]
                and m7[i] > m7[i - 1]
                and ha["close"].iloc[i] > tenkan[i]
            )
            if valide:
                statuts[i] = f"FERNANDA (stromboli du {ha.index[actif_haussier].date()})"
                actif_haussier = None
            elif ha["low"].iloc[i] < ha["low"].iloc[i - 1]:
                statuts[i] = "invalidation (cassure du plus bas de la bougie precedente)"
                actif_haussier = None

    debut = max(0, len(ha) - nb_bougies)
    print(
        f"\n{'Date':<12}{'O':>9}{'H':>9}{'L':>9}{'C':>9}"
        f"{'M7':>9}{'Tenkan':>9}  Statut"
    )
    print("-" * 90)
    for i in range(debut, len(ha)):
        rouge = est_rouge_pleine(ha, i)
        doji = est_doji(ha, i)
        marque = "R" if rouge else ("D" if doji else " ")
        m7_str = f"{m7[i]:.2f}" if not np.isnan(m7[i]) else "  n/a"
        tenkan_str = f"{tenkan[i]:.2f}" if not np.isnan(tenkan[i]) else "  n/a"
        print(
            f"{ha.index[i].date()!s:<12}"
            f"{ha['open'].iloc[i]:>9.2f}{ha['high'].iloc[i]:>9.2f}"
            f"{ha['low'].iloc[i]:>9.2f}{ha['close'].iloc[i]:>9.2f}"
            f"{m7_str:>9}{tenkan_str:>9}  [{marque}] {statuts.get(i, '')}"
        )
    print("\n[R] = rouge pleine (HA_high == HA_open)   [D] = doji")


TWELVEDATA_API = "https://api.twelvedata.com"
TWELVEDATA_EXCHANGE = {".PA": "XPAR", ".AS": "XAMS", ".BR": "XBRU"}


def _exchange_twelvedata(ticker):
    """Devine le parametre 'exchange' Twelve Data a partir du suffixe du ticker."""
    for suffixe, code in TWELVEDATA_EXCHANGE.items():
        if ticker.endswith(suffixe):
            return code
    return None  # US et autres : pas de parametre exchange necessaire


def telecharger_twelvedata(tickers, outputsize=700):
    """
    Filet de secours utilise UNIQUEMENT pour les tickers que yfinance n'a pas
    reussi a recuperer. Ne remplace jamais Yahoo comme source principale :
    le plan gratuit de Twelve Data ne couvre que les marches US, pas
    Euronext (Paris/Amsterdam/Bruxelles necessitent un palier payant chez
    eux) — donc ce filet n'aidera reellement que sur les echecs US.

    Retourne {ticker: DataFrame} au meme format que telecharger().
    """
    if not TWELVEDATA_API_KEY or not tickers:
        return {}

    donnees = {}
    for ticker in tickers:
        symbole = ticker.split(".")[0]  # Twelve Data attend le symbole nu
        exchange = _exchange_twelvedata(ticker)
        params = {
            "symbol": symbole,
            "interval": "1day",
            "outputsize": outputsize,
            "apikey": TWELVEDATA_API_KEY,
        }
        if exchange:
            params["exchange"] = exchange

        try:
            reponse = requests.get(f"{TWELVEDATA_API}/time_series", params=params, timeout=15)
            reponse.raise_for_status()
            data = reponse.json()

            if data.get("status") == "error" or "values" not in data:
                continue

            valeurs = data["values"]
            if len(valeurs) < MIN_BOUGIES + 25:
                continue

            cadre = pd.DataFrame(valeurs)
            cadre["datetime"] = pd.to_datetime(cadre["datetime"])
            cadre = cadre.set_index("datetime").sort_index()
            cadre = cadre.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            colonnes = ["Open", "High", "Low", "Close"] + (["Volume"] if "volume" in valeurs[0] else [])
            cadre = cadre[colonnes].astype(float)
            if "Volume" not in cadre.columns:
                cadre["Volume"] = 0.0

            donnees[ticker] = cadre
        except Exception:
            continue

        time.sleep(8)  # reste sous la limite gratuite de 8 requetes/minute

    return donnees


def _periode_en_jours(periode):
    """Convertit '2y'/'1y'/'3mo'/'5y' en nombre de jours pour construire la date 'from' EODHD."""
    if periode.endswith("y"):
        return int(periode[:-1]) * 365
    if periode.endswith("mo"):
        return int(periode[:-2]) * 31
    return 730  # defaut ~2 ans


def _symbole_eodhd(ticker):
    """
    Convertit un ticker interne en symbole EODHD. Euronext (.PA/.AS/.BR)
    et les indices (.INDX) gardent leur format, deja natif EODHD. Les
    tickers US sans suffixe (ex: 'AAPL') recoivent '.US', requis par EODHD.
    """
    if any(ticker.endswith(s) for s in (".PA", ".AS", ".BR", ".INDX")):
        return ticker
    return f"{ticker}.US"


def telecharger(tickers, periode):
    """
    Telecharge l'historique daily via EODHD (source principale pour
    actions US/Euronext/Indices depuis la migration hors Yahoo Finance).
    Une requete par ticker (EODHD n'a pas de mode batch sur ce palier).
    Retourne {ticker: DataFrame OHLCV}.
    """
    donnees = {}
    if not EODHD_API_KEY:
        print("  EODHD_API_KEY manquante : aucun telechargement possible.")
        return donnees

    depuis = (datetime.now(timezone.utc) - pd.Timedelta(days=_periode_en_jours(periode))).strftime("%Y-%m-%d")

    for i, ticker in enumerate(tickers, 1):
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)} tickers EODHD...", flush=True)

        try:
            reponse = requests.get(
                f"{EODHD_API}/eod/{_symbole_eodhd(ticker)}",
                params={"api_token": EODHD_API_KEY, "fmt": "json", "from": depuis, "period": "d"},
                timeout=15,
            )
            reponse.raise_for_status()
            valeurs = reponse.json()

            if not isinstance(valeurs, list) or len(valeurs) < MIN_BOUGIES + 25:
                continue

            cadre = pd.DataFrame(valeurs)
            cadre["date"] = pd.to_datetime(cadre["date"])
            cadre = cadre.set_index("date").sort_index()

            # EODHD ne fournit que la cloture ajustee (adjusted_close), jamais
            # les equivalents pour Open/High/Low. Sans correction, un split
            # (ex: 10-pour-1) cree un ecart artificiel de plusieurs dizaines
            # de % entre la cloture ajustee de la veille et l'ouverture brute
            # du lendemain — invisible sur un backtest close-to-close, mais
            # catastrophique des qu'on compare a un stop ou qu'on simule une
            # entree a l'ouverture. On applique le meme facteur d'ajustement
            # (cloture ajustee / cloture brute) a Open/High/Low, comme le
            # fait n'importe quel fournisseur serieux en interne.
            if "close" in cadre.columns and "adjusted_close" in cadre.columns:
                facteur = cadre["adjusted_close"] / cadre["close"].replace(0, np.nan)
                cadre["Open"] = cadre["open"] * facteur
                cadre["High"] = cadre["high"] * facteur
                cadre["Low"] = cadre["low"] * facteur
                cadre["Close"] = cadre["adjusted_close"]
                cadre["Volume"] = cadre["volume"]
            else:
                cadre = cadre.rename(columns={
                    "open": "Open", "high": "High", "low": "Low",
                    "adjusted_close": "Close", "volume": "Volume",
                })

            cadre = cadre[["Open", "High", "Low", "Close", "Volume"]].dropna(
                subset=["Open", "High", "Low", "Close"]
            ).astype(float)

            if len(cadre) >= MIN_BOUGIES + 25:
                donnees[ticker] = cadre
        except Exception:
            continue

        time.sleep(0.05)  # tres large marge sous la limite EODHD de 1000/min

    # Filet de secours Twelve Data, seulement pour ce qui manque encore et
    # seulement si une cle est configuree (desactive par defaut, aucun
    # changement de comportement si TWELVEDATA_API_KEY n'est pas definie).
    if TWELVEDATA_API_KEY:
        manquants = [t for t in tickers if t not in donnees]
        if manquants:
            print(f"  {len(manquants)} tickers absents d'EODHD, tentative via Twelve Data...")
            recuperes = telecharger_twelvedata(manquants)
            if recuperes:
                print(f"  {len(recuperes)} recuperes via Twelve Data : {sorted(recuperes.keys())}")
                donnees.update(recuperes)

    return donnees


def telecharger_kraken(tickers, periode):
    """
    Telecharge l'historique daily de plusieurs paires Kraken (une requete
    par paire, l'API Kraken n'a pas de mode batch). Retourne un dict
    {ticker: DataFrame} au MEME format (colonnes Open/High/Low/Close/Volume,
    index datetime) que telecharger(), pour rester compatible avec toute
    la chaine de traitement en aval (heikin_ashi, detection, backtest...).
    """
    annees = int(periode.rstrip("y")) if periode.endswith("y") else 2
    depuis = int((datetime.now(timezone.utc) - pd.Timedelta(days=annees * 365)).timestamp())

    donnees = {}
    for i, ticker in enumerate(tickers, 1):
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} paires Kraken...", flush=True)
        try:
            reponse = requests.get(
                f"{KRAKEN_API}/OHLC",
                params={"pair": ticker, "interval": 1440, "since": depuis},
                timeout=15,
            )
            reponse.raise_for_status()
            data = reponse.json()
            if data.get("error"):
                continue

            resultat = data.get("result", {})
            cles = [c for c in resultat if c != "last"]
            if not cles:
                continue

            lignes = resultat[cles[0]]
            if len(lignes) < MIN_BOUGIES + 25:
                continue

            cadre = pd.DataFrame(
                lignes,
                columns=["time", "Open", "High", "Low", "Close", "vwap", "Volume", "count"],
            )
            cadre["time"] = pd.to_datetime(cadre["time"], unit="s")
            cadre = cadre.set_index("time")[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            donnees[ticker] = cadre
        except Exception:
            continue

        time.sleep(0.3)

    return donnees


def telecharger_pour_place(place, tickers, periode):
    """Aiguille vers le bon telechargeur selon la place (Kraken pour Crypto)."""
    if place == "Crypto":
        return telecharger_kraken(tickers, periode)
    return telecharger(tickers, periode)


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scanner(univers, timeframes, periode=PERIODE_DAILY):
    """Scan de la derniere bougie cloturee. Retourne la liste des signaux."""
    signaux = []

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, ohlc in donnees.items():
            for tf in timeframes:
                cadre = agreger_tf(ohlc, tf)
                if len(cadre) < MIN_BOUGIES + 2:
                    continue

                ha = heikin_ashi(cadre)
                trouve = detecter_stromboli(ha, len(ha) - 1)
                if trouve:
                    trouve["ticker"] = ticker
                    trouve["place"] = place
                    trouve["tf"] = tf
                    trouve["type"] = "stromboli"
                    signaux.append(trouve)
                    print(f"  >> STROMBOLI {tf} {trouve['sens']} : {ticker}")

                occurrences = detecter_fernanda_series(ha)
                if occurrences and occurrences[-1]["index"] == len(ha) - 1:
                    fern = occurrences[-1]
                    signal_fern = {
                        "type": fern["type"],
                        "ticker": ticker,
                        "place": place,
                        "tf": tf,
                        "date": fern["date"],
                        "ha_close": float(ha["close"].iloc[-1]),
                        "stromboli_date": fern["stromboli_date"],
                    }
                    if "volume" in ha.columns:
                        i = len(ha) - 1
                        volume = float(ha["volume"].iloc[i])
                        debut = max(0, i - 20)
                        moyenne = float(ha["volume"].iloc[debut:i].mean()) if i > debut else 0.0
                        signal_fern["volume_ratio"] = volume / moyenne if moyenne > 0 else None
                    signaux.append(signal_fern)
                    print(f"  >> {fern['type'].upper()} {tf} : {ticker}")

    return signaux


def scanner_historique(univers, timeframes, annees):
    """Compte tous les Stromboli de l'historique. Pour calibrer les seuils."""
    lignes = []
    periode = f"{annees}y"

    for place, tickers in univers.items():
        print(f"\n[{place}] telechargement de {len(tickers)} tickers ({periode})")
        donnees = telecharger_pour_place(place, tickers, periode)
        print(f"  {len(donnees)} tickers exploitables")

        for ticker, ohlc in donnees.items():
            for tf in timeframes:
                cadre = agreger_tf(ohlc, tf)
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
        donnees = telecharger_pour_place(place, tickers, "3mo")
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
        envoye = False
        for tentative in range(3):
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
                if reponse.status_code == 200:
                    envoye = True
                    break
                print(f"Telegram HTTP {reponse.status_code} : {reponse.text[:200]}")
            except Exception as erreur:
                print(f"Telegram erreur (tentative {tentative + 1}/3) : {erreur}")
                time.sleep(3 * (tentative + 1))
        if not envoye:
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


ETIQUETTES_TF = {"D": "DAILY", "W": "WEEKLY", "M": "MONTHLY"}


def formater(signaux, timeframes, titre="STROMBOLI"):
    horodatage = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    stromboli = [s for s in signaux if s.get("type") == "stromboli"]
    fernanda = [s for s in signaux if s.get("type") in ("fernanda", "fernando")]

    if not signaux:
        return f"<b>{titre}</b> — {horodatage}\n\nAucun signal sur {', '.join(timeframes)}."

    lignes = [f"<b>{titre}</b> — {horodatage}", ""]

    for tf in timeframes:
        groupe = [s for s in stromboli if s["tf"] == tf]
        if not groupe:
            continue

        etiquette = ETIQUETTES_TF.get(tf, tf)
        lignes.append(f"<b>▲ {etiquette} HAUSSIER</b> ({len(groupe)})")

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

    for tf in timeframes:
        groupe = [s for s in fernanda if s["tf"] == tf]
        if not groupe:
            continue

        etiquette = ETIQUETTES_TF.get(tf, tf)
        lignes.append(f"<b>▲ {etiquette} FERNANDA</b> ({len(groupe)})")

        for signal in sorted(groupe, key=lambda s: s["ticker"]):
            date_strom = signal["stromboli_date"].strftime("%d/%m")
            detail = f"stromboli du {date_strom}"
            ratio = signal.get("volume_ratio")
            if ratio:
                detail += f" · vol x{ratio:.1f}"
            lignes.append(
                f"  <code>{signal['ticker']}</code> — {signal['ha_close']:.2f}\n"
                f"     {detail}"
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
    parseur.add_argument("--tf", default="DW", help="combinaison de D, W, M (ex: DWM, defaut DW)")
    parseur.add_argument(
        "--univers",
        default="tout",
        choices=["tout", "us", "euronext", "paris", "amsterdam", "bruxelles", "indices", "crypto"],
    )
    parseur.add_argument("--dry-run", action="store_true", help="pas d'envoi Telegram")
    parseur.add_argument("--valider-univers", action="store_true")
    parseur.add_argument("--historique", type=int, metavar="ANNEES")
    parseur.add_argument(
        "--backtest", type=int, metavar="ANNEES",
        help="probabilite de reussite des Fernanda (rendement reel, plusieurs horizons)",
    )
    parseur.add_argument(
        "--volume-min", type=float, metavar="RATIO", default=None,
        help="filtre d'analyse du backtest : garde uniquement les doji avec "
             "volume >= RATIO fois leur moyenne 20 bougies (jamais utilise en scan reel)",
    )
    parseur.add_argument(
        "--avec-fernando", action="store_true",
        help="backtest uniquement : calcule aussi le miroir baissier (Stromboli baissier "
             "+ Fernando/short), jamais utilise en scan reel ni alertes Telegram",
    )
    parseur.add_argument(
        "--setup", default="inchi", choices=["inchi", "retour-moyenne", "ema-cross", "ema-cross-partiel"],
        help="backtest uniquement : 'inchi' (Stromboli/Fernanda, defaut), "
             "'retour-moyenne' (RSI-2 / IBS + filtre MM200), "
             "'ema-cross' (suivi de tendance EMA 8/21, Chandelier Exit) ou "
             "'ema-cross-partiel' (meme entree, moitie sortie tot + moitie Chandelier)",
    )
    parseur.add_argument(
        "--stop-loss", type=float, metavar="PCT", default=None,
        help="setup retour-moyenne uniquement : stop loss en %% (ex: -8 pour -8%%), "
             "sortie immediate si le plus bas de seance le franchit",
    )
    parseur.add_argument(
        "--frais-pct", type=float, metavar="PCT", default=0.0,
        help="setup retour-moyenne uniquement : frais de courtage aller-retour en %% de "
             "la position (ex: 0.2 pour 2 euros sur 1000 euros), 0 = aucun frais (defaut). "
             "A calibrer selon TON compte reel, jamais suppose par le code.",
    )
    parseur.add_argument(
        "--volume-min-signal", type=float, metavar="RATIO", default=None,
        help="setup retour-moyenne uniquement : garde seulement les signaux avec volume du "
             "jour >= RATIO fois sa moyenne 20 jours (pic de volume au signal)",
    )
    parseur.add_argument(
        "--volume-min-liquidite", type=float, metavar="VOLUME", default=None,
        help="setup retour-moyenne uniquement : garde seulement les signaux dont le volume "
             "moyen des 60 jours precedents est >= VOLUME titres/jour (liquidite generale)",
    )
    parseur.add_argument(
        "--entree-lendemain", action="store_true",
        help="setup retour-moyenne uniquement : simule une entree REALISTE a l'ouverture du "
             "jour suivant le signal, plutot qu'a la cloture du jour du signal (optimiste)",
    )
    parseur.add_argument(
        "--validation-croisee", action="store_true",
        help="setup retour-moyenne uniquement : decoupe les trades en decouverte/validation "
             "(coupure = date mediane) pour verifier que l'edge tient hors echantillon",
    )
    parseur.add_argument(
        "--diagnostic", metavar="TICKER",
        help="affiche les valeurs HA/M7/Tenkan bougie par bougie pour un ticker (ex: ELI.BR)",
    )
    args = parseur.parse_args()

    timeframes = [c for c in "DWM" if c in args.tf.upper()]
    if not timeframes:
        print("--tf doit contenir D, W et/ou M")
        return 1

    if args.diagnostic:
        for tf in timeframes:
            print(f"\n{'=' * 20} {args.diagnostic} — {tf} {'=' * 20}")
            diagnostiquer(args.diagnostic, tf)
        return 0

    print("=" * 60)
    print("BOT STROMBOLI — methode Inchi")
    print(
        f"Parametres : {MIN_BOUGIES} bougies min · doji <= {SEUIL_DOJI * 100:.0f}% "
        f"· tolerance meche {TOLERANCE_MECHE * 100:.2f}%"
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

    if args.backtest:
        if args.setup == "ema-cross":
            resultats = backtest_ema_cross(
                univers, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct
            )
            print("\n" + resume_ema_cross(
                resultats, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct
            ))
            if args.validation_croisee:
                print("\n" + resume_validation_croisee(resultats, args.backtest, stop_pct=args.stop_loss))
            for d, df in resultats.items():
                if not df.empty:
                    chemin = RACINE / f"backtest_{d}.csv"
                    df.to_csv(chemin, index=False)
                    print(f"Detail {d} ecrit dans {chemin.name}")
            return 0

        if args.setup == "ema-cross-partiel":
            resultats = backtest_ema_cross_partiel(
                univers, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct
            )
            print("\n" + resume_ema_cross_partiel(
                resultats, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct
            ))
            if args.validation_croisee:
                print("\n" + resume_validation_croisee(resultats, args.backtest, stop_pct=args.stop_loss))
            for d, df in resultats.items():
                if not df.empty:
                    chemin = RACINE / f"backtest_{d}.csv"
                    df.to_csv(chemin, index=False)
                    print(f"Detail {d} ecrit dans {chemin.name}")
            return 0

        if args.setup == "retour-moyenne":
            resultats = backtest_retour_moyenne(
                univers, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct,
                volume_min_signal=args.volume_min_signal, volume_min_liquidite=args.volume_min_liquidite,
                entree_lendemain=args.entree_lendemain,
            )
            print("\n" + resume_retour_moyenne(
                resultats, args.backtest, stop_pct=args.stop_loss, frais_pct=args.frais_pct,
                volume_min_signal=args.volume_min_signal, volume_min_liquidite=args.volume_min_liquidite,
                entree_lendemain=args.entree_lendemain,
            ))
            if args.validation_croisee:
                print("\n" + resume_validation_croisee(resultats, args.backtest, stop_pct=args.stop_loss))
            for d, df in resultats.items():
                if not df.empty:
                    chemin = RACINE / f"backtest_rm_{d}.csv"
                    df.to_csv(chemin, index=False)
                    print(f"Detail {d} ecrit dans {chemin.name}")
            return 0

        if args.avec_fernando:
            df_stromboli, df_fernanda, df_fernando = backtest_fernanda(
                univers, args.backtest, volume_min=args.volume_min, avec_fernando=True
            )
        else:
            df_stromboli, df_fernanda = backtest_fernanda(univers, args.backtest, volume_min=args.volume_min)
            df_fernando = None

        rapport = resume_backtest(
            df_stromboli, df_fernanda, args.backtest,
            volume_min=args.volume_min, df_fernando=df_fernando,
        )
        print("\n" + rapport)
        if not df_stromboli.empty:
            chemin = RACINE / "backtest_stromboli.csv"
            df_stromboli.to_csv(chemin, index=False)
            print(f"\nDetail Stromboli ecrit dans {chemin.name}")
        if not df_fernanda.empty:
            chemin = RACINE / "backtest_fernanda.csv"
            df_fernanda.to_csv(chemin, index=False)
            print(f"Detail Fernanda ecrit dans {chemin.name}")
        if df_fernando is not None and not df_fernando.empty:
            chemin = RACINE / "backtest_fernando.csv"
            df_fernando.to_csv(chemin, index=False)
            print(f"Detail Fernando ecrit dans {chemin.name}")
        return 0

    signaux = scanner(univers, timeframes)
    print(f"\n{len(signaux)} signal(aux) detecte(s)")

    # Deux notifications separees : Crypto d'un cote, tout le reste (actions,
    # indices) de l'autre. On n'envoie une notif pour un groupe que si ce
    # groupe faisait bien partie du scan demande (evite un message "Aucun
    # signal" superflu quand on lance un scan cible, ex --univers crypto).
    a_crypto = "Crypto" in univers
    a_actions = any(place != "Crypto" for place in univers)

    if a_crypto:
        signaux_crypto = [s for s in signaux if s["place"] == "Crypto"]
        envoyer_telegram(
            formater(signaux_crypto, timeframes, titre="STROMBOLI CRYPTO"),
            dry_run=args.dry_run,
        )

    if a_actions:
        signaux_actions = [s for s in signaux if s["place"] != "Crypto"]
        envoyer_telegram(
            formater(signaux_actions, timeframes, titre="STROMBOLI ACTIONS"),
            dry_run=args.dry_run,
        )

    resume = os.getenv("GITHUB_STEP_SUMMARY")
    if resume:
        with open(resume, "a", encoding="utf-8") as fichier:
            fichier.write(f"## Stromboli — {len(signaux)} signal(aux)\n\n")
            for signal in signaux:
                if signal["type"] == "stromboli":
                    fichier.write(
                        f"- `{signal['ticker']}` {signal['tf']} stromboli {signal['sens']} "
                        f"({signal['bougies']} bougies)\n"
                    )
                else:
                    fichier.write(
                        f"- `{signal['ticker']}` {signal['tf']} {signal['type']}\n"
                    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
