#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stromboli.py — Bot d'alerte Stromboli (methode Inchi)

Detecte les figures Stromboli en Heikin Ashi, sur les unites de temps
Daily et Weekly, sur les actions US (Nasdaq 100 + S&P 500) et Euronext
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
    Petite liste curee d'indices/futures, pas de decouverte automatique
    (volume trop faible pour justifier une source dynamique).

    US : futures continus (bien couverts par Yahoo, roulement automatique).
    Europe : indices CASH plutot que futures. Les futures Euronext/Eurex
    (FCE, FDAX, FESX) n'ont pas de serie continue fiable sur Yahoo Finance
    car les contrats expirent chaque trimestre avec un nouveau code. L'indice
    cash suit le future de tres pres (arbitrage), donc c'est un proxy fiable
    pour la detection Stromboli/Fernanda.
    """
    return [
        "ES=F", "NQ=F", "YM=F", "RTY=F",   # US : S&P500, Nasdaq, Dow, Russell2000
        "^GDAXI", "^FCHI", "^STOXX50E",     # Europe : DAX, CAC40, Euro Stoxx 50
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
    """
    mensuel = ohlc.resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Open", "Close"])

    if len(mensuel) == 0:
        return mensuel

    dernier_jour = ohlc.index[-1]
    if mensuel.index[-1] > dernier_jour:
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


def backtest_fernanda(univers, annees, horizons=HORIZONS_BACKTEST, volume_min=None):
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

    Retourne (DataFrame Stromboli, DataFrame Fernanda), une ligne par signal.
    """
    periode = f"{annees}y"
    lignes_stromboli = []
    lignes_fernanda = []

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

            def rendements(i):
                prix_entree = closes_reels[i]
                ligne = {"prix_entree": prix_entree}
                for h in horizons:
                    j = i + h
                    if j < n and prix_entree > 0:
                        ligne[f"rendement_{h}j"] = (closes_reels[j] - prix_entree) / prix_entree * 100
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


def resume_backtest(df_stromboli, df_fernanda, annees, volume_min=None, horizons=HORIZONS_BACKTEST):
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

    if not df_fernanda.empty:
        sortie.append("Fernanda par place :")
        sortie.append(str(df_fernanda.groupby("place").size().rename("signaux")))

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

    cadre = donnees[ticker] if tf == "D" else to_weekly(donnees[ticker])
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
        "--diagnostic", metavar="TICKER",
        help="affiche les valeurs HA/M7/Tenkan bougie par bougie pour un ticker (ex: ELI.BR)",
    )
    args = parseur.parse_args()

    timeframes = [c for c in "DWM" if c in args.tf.upper()]
    if not timeframes:
        print("--tf doit contenir D et/ou W")
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

    if args.backtest:
        df_stromboli, df_fernanda = backtest_fernanda(univers, args.backtest, volume_min=args.volume_min)
        rapport = resume_backtest(df_stromboli, df_fernanda, args.backtest, volume_min=args.volume_min)
        print("\n" + rapport)
        if not df_stromboli.empty:
            chemin = RACINE / "backtest_stromboli.csv"
            df_stromboli.to_csv(chemin, index=False)
            print(f"\nDetail Stromboli ecrit dans {chemin.name}")
        if not df_fernanda.empty:
            chemin = RACINE / "backtest_fernanda.csv"
            df_fernanda.to_csv(chemin, index=False)
            print(f"Detail Fernanda ecrit dans {chemin.name}")
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
