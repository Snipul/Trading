import numpy as np
import pandas as pd
import stromboli as S

echecs = []


def verifier(nom, condition):
    print(("  OK   " if condition else "  ECHEC") + f"  {nom}")
    if not condition:
        echecs.append(nom)


# --- 1. Formule Heikin Ashi -------------------------------------------------
print("\n1. Formule Heikin Ashi")
ohlc = pd.DataFrame(
    {"Open": [100.0, 98.0], "High": [101.0, 99.0],
     "Low": [97.0, 95.0], "Close": [98.0, 96.0], "Volume": [1000.0, 1200.0]},
    index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
)
ha = S.heikin_ashi(ohlc)
verifier("HA_close = (O+H+L+C)/4", np.isclose(ha["close"].iloc[0], 99.0))
verifier("HA_open initial = (O+C)/2", np.isclose(ha["open"].iloc[0], 99.0))
verifier("HA_open suivant = moyenne des deux precedents",
         np.isclose(ha["open"].iloc[1], (99.0 + 99.0) / 2))
verifier("HA_high = max(H, HA_open, HA_close)", np.isclose(ha["high"].iloc[0], 101.0))
verifier("HA_low = min(L, HA_open, HA_close)", np.isclose(ha["low"].iloc[0], 97.0))


# --- 2. Primitives sur bougies fabriquees -----------------------------------
print("\n2. Reconnaissance des bougies")


def cadre_ha(bougies):
    index = pd.date_range("2024-01-01", periods=len(bougies), freq="D")
    return pd.DataFrame(
        bougies, columns=["open", "high", "low", "close"], index=index
    ).assign(volume=1_000_000.0)


# rouge pleine : high == open, cloture plus bas
rouge_pleine = cadre_ha([[100.0, 100.0, 96.0, 97.0]])
verifier("rouge pleine reconnue", S.est_rouge_pleine(rouge_pleine, 0))

# rouge avec une meche haute minuscule -> rejetee (tolerance zero)
rouge_meche = cadre_ha([[100.0, 100.01, 96.0, 97.0]])
verifier("rouge avec micro-meche rejetee", not S.est_rouge_pleine(rouge_meche, 0))

# verte pleine : low == open
verte_pleine = cadre_ha([[100.0, 104.0, 100.0, 103.0]])
verifier("verte pleine reconnue", S.est_verte_pleine(verte_pleine, 0))
verte_meche = cadre_ha([[100.0, 104.0, 99.99, 103.0]])
verifier("verte avec micro-meche rejetee", not S.est_verte_pleine(verte_meche, 0))

# doji : corps 2% du range, meches des deux cotes
doji = cadre_ha([[100.0, 102.0, 98.0, 100.05]])
verifier("doji reconnu", S.est_doji(doji, 0))

# corps trop gros
gros = cadre_ha([[100.0, 102.0, 98.0, 101.5]])
verifier("corps trop gros rejete", not S.est_doji(gros, 0))

# petit corps mais sans meche basse -> pas un doji
sans_meche = cadre_ha([[100.0, 102.0, 100.0, 100.05]])
verifier("petit corps sans meche basse rejete", not S.est_doji(sans_meche, 0))

# doji dragonfly : pas de meche haute, meche basse presente
dragonfly = cadre_ha([[100.0, 100.0, 96.0, 100.0]])
verifier(
    "dragonfly rejete par defaut (INCLURE_DRAGONFLY=False)",
    S.INCLURE_DRAGONFLY == False and not S.est_doji(dragonfly, 0),
)
gravestone = cadre_ha([[100.0, 104.0, 100.0, 100.0]])
S.INCLURE_DRAGONFLY = True
verifier("dragonfly accepte quand INCLURE_DRAGONFLY=True", S.est_doji(dragonfly, 0))
verifier("gravestone jamais accepte (baissier desactive)", not S.est_doji(gravestone, 0))
verifier("doji classique toujours accepte avec l'option activee", S.est_doji(doji, 0))
S.INCLURE_DRAGONFLY = False  # restaure l'etat par defaut pour la suite des tests


# --- 3. Detection du Stromboli ---------------------------------------------
print("\n3. Detection du Stromboli")

haussier = cadre_ha([
    [110.0, 110.0, 106.0, 107.0],
    [107.0, 107.0, 103.0, 104.0],
    [104.0, 104.0, 100.0, 101.0],
    [100.0, 102.0, 98.0, 100.05],
])
resultat = S.detecter_stromboli(haussier, 3)
verifier("3 rouges + doji -> stromboli", resultat is not None)
verifier("sens haussier", resultat and resultat["sens"] == "haussier")
verifier("3 bougies comptees", resultat and resultat["bougies"] == 3)

baissier = cadre_ha([
    [100.0, 104.0, 100.0, 103.0],
    [103.0, 107.0, 103.0, 106.0],
    [106.0, 110.0, 106.0, 109.0],
    [110.0, 112.0, 108.0, 110.05],
])
resultat = S.detecter_stromboli(baissier, 3)
verifier("3 vertes + doji -> aucun signal (baissier desactive)", resultat is None)

# seulement 2 rouges -> rien
deux_rouges = cadre_ha([
    [110.0, 112.0, 106.0, 107.0],   # rouge AVEC meche haute, casse la serie
    [107.0, 107.0, 103.0, 104.0],
    [104.0, 104.0, 100.0, 101.0],
    [100.0, 102.0, 98.0, 100.05],
])
verifier("2 rouges pleines seulement -> aucun signal",
         S.detecter_stromboli(deux_rouges, 3) is None)

# 5 rouges d'affilee -> serie longue comptee
cinq = cadre_ha([
    [119.0, 119.0, 115.0, 116.0],
    [116.0, 116.0, 112.0, 113.0],
    [113.0, 113.0, 109.0, 110.0],
    [110.0, 110.0, 106.0, 107.0],
    [107.0, 107.0, 103.0, 104.0],
    [104.0, 106.0, 102.0, 104.05],
])
resultat = S.detecter_stromboli(cinq, 5)
verifier("serie de 5 comptee correctement", resultat and resultat["bougies"] == 5)

# doji non contigu (une bougie neutre s'intercale)
non_contigu = cadre_ha([
    [110.0, 110.0, 106.0, 107.0],
    [107.0, 107.0, 103.0, 104.0],
    [104.0, 104.0, 100.0, 101.0],
    [101.0, 105.0, 100.0, 104.0],   # verte, casse la contiguite
    [104.0, 106.0, 102.0, 104.05],
])
verifier("doji non contigu -> aucun signal",
         S.detecter_stromboli(non_contigu, 4) is None)


# --- 6. Fernanda / Fernando -------------------------------------------------
print("\n4. Fernanda / Fernando")


def scenario_fernanda(corps_doji=0.1, sens="haussier"):
    """Filler (9) + 3 bougies pleines + doji + reprise franche dans le sens oppose."""
    index = pd.date_range("2024-01-01", periods=25, freq="D")
    lignes = [[100.0, 101.0, 99.0, 100.0] for _ in range(9)]

    if sens == "haussier":
        lignes += [
            [100.0, 100.0, 96.0, 97.0],
            [97.0, 97.0, 93.0, 94.0],
            [94.0, 94.0, 90.0, 91.0],
        ]
        lignes.append([91.0, 93.0, 89.0, 91.0 + corps_doji])
        for k in range(12):
            base = 92 + k * 4
            lignes.append([base, base + 6, base - 1, base + 5])
    else:
        lignes += [
            [100.0, 104.0, 100.0, 103.0],
            [103.0, 107.0, 103.0, 106.0],
            [106.0, 110.0, 106.0, 109.0],
        ]
        lignes.append([109.0, 111.0, 107.0, 109.0 - corps_doji])
        for k in range(12):
            base = 108 - k * 4
            lignes.append([base, base + 1, base - 6, base - 5])

    ha = pd.DataFrame(lignes, columns=["open", "high", "low", "close"], index=index[: len(lignes)])
    ha["volume"] = 1_000_000.0
    return ha


ha_haussier = scenario_fernanda(sens="haussier")
occurrences = S.detecter_fernanda_series(ha_haussier)
verifier("fernanda detectee apres stromboli haussier", len(occurrences) >= 1)
verifier(
    "fernanda posterieure au doji (index 12)",
    occurrences and occurrences[0]["type"] == "fernanda" and occurrences[0]["index"] > 12,
)

ha_baissier = scenario_fernanda(sens="baissier")
occurrences_b = S.detecter_fernanda_series(ha_baissier)
verifier(
    "aucun fernando (baissier desactive, pas de stromboli baissier source)",
    len(occurrences_b) == 0,
)

# calcul_m7 / calcul_tenkan : verification directe sur une serie simple
serie_simple = cadre_ha([[100.0, 101.0, 99.0, 100.0 + i] for i in range(10)])
m7 = S.calcul_m7(serie_simple)
verifier("m7 = NaN avant 7 bougies", np.isnan(m7[5]))
verifier(
    "m7 correcte a l'indice 6",
    np.isclose(m7[6], serie_simple["close"].iloc[0:7].mean()),
)

tenkan = S.calcul_tenkan(serie_simple)
verifier("tenkan = NaN avant 9 bougies", np.isnan(tenkan[7]))
verifier(
    "tenkan correcte a l'indice 9",
    np.isclose(
        tenkan[9],
        (serie_simple["high"].iloc[1:10].max() + serie_simple["low"].iloc[1:10].min()) / 2,
    ),
)

# Pas de re-signal : une fois la Fernanda declenchee, le meme Stromboli
# ne doit pas re-emettre tant qu'aucun nouveau Stromboli n'apparait.
dates_fernanda = [o["stromboli_date"] for o in occurrences if o["type"] == "fernanda"]
verifier(
    "un seul stromboli source pour la fernanda detectee",
    len(set(dates_fernanda)) == len(dates_fernanda),
)

# Invalidation : une bougie qui fait un nouveau plus bas HA juste apres le
# doji doit desactiver la surveillance sans emettre de fernanda.
index_inv = pd.date_range("2024-02-01", periods=14, freq="D")
lignes_inv = [[100.0, 101.0, 99.0, 100.0] for _ in range(9)]
lignes_inv += [
    [100.0, 100.0, 96.0, 97.0],
    [97.0, 97.0, 93.0, 94.0],
    [94.0, 94.0, 90.0, 91.0],
]
lignes_inv.append([91.0, 93.0, 89.0, 91.1])  # doji, low=89
lignes_inv.append([91.0, 92.0, 87.0, 88.0])  # nouveau plus bas -> invalidation
ha_inv = pd.DataFrame(
    lignes_inv, columns=["open", "high", "low", "close"], index=index_inv[: len(lignes_inv)]
)
ha_inv["volume"] = 1_000_000.0
occ_inv = S.detecter_fernanda_series(ha_inv)
verifier("invalidation : aucune fernanda apres cassure du plus bas", len(occ_inv) == 0)



# --- 5. Backtest Fernanda ---------------------------------------------------
print("\n5. Backtest Fernanda")

ha_bt = scenario_fernanda(sens="haussier")
occurrences_bt = S.detecter_fernanda_series(ha_bt)
verifier("scenario backtest : une fernanda presente", len(occurrences_bt) == 1)

i_entree = occurrences_bt[0]["index"]
n = len(ha_bt)
closes_reels = np.linspace(100.0, 100.0 + (n - 1) * 2, n)  # hausse lineaire connue
prix_entree = closes_reels[i_entree]

for h in (1, 3, 5):
    j = i_entree + h
    if j < n:
        rendement_attendu = (closes_reels[j] - prix_entree) / prix_entree * 100
        verifier(
            f"rendement a {h}j coherent avec une hausse lineaire connue",
            rendement_attendu > 0,
        )

df_bt = pd.DataFrame([{
    "ticker": "TEST", "place": "US", "date": ha_bt.index[i_entree],
    "prix_entree": prix_entree,
    "rendement_1j": 2.0, "rendement_3j": 5.0, "rendement_5j": 8.0,
    "rendement_10j": None, "rendement_20j": None,
}])
df_bt_strom = pd.DataFrame([{
    "ticker": "TEST", "place": "US", "date": ha_bt.index[12],
    "prix_entree": prix_entree,
    "rendement_1j": -1.0, "rendement_3j": 1.0, "rendement_5j": 3.0,
    "rendement_10j": None, "rendement_20j": None,
}])
rapport = S.resume_backtest(df_bt_strom, df_bt, annees=1, horizons=(1, 3, 5, 10, 20))
verifier("rapport backtest : taux 100% affiche pour la fernanda", "100.0%" in rapport)
verifier("rapport backtest : horizons sans donnee absents", "10j" not in rapport)
verifier("rapport backtest : taux de validation affiche", "validation" in rapport)
verifier("rapport backtest : section entree directe presente", "ENTREE DIRECTE" in rapport)
verifier("rapport backtest : section fernanda presente", "ENTREE A LA FERNANDA" in rapport)

vide = S.resume_backtest(pd.DataFrame(), pd.DataFrame(), annees=1)
verifier("rapport backtest vide gere", "aucun signal exploitable" in vide)
verifier("rapport backtest vide : 0 stromboli affiche", "Stromboli detectes : 0" in vide)


# --- 6. Agregation hebdomadaire --------------------------------------------
print("\n6. Agregation hebdomadaire")
jours = pd.date_range("2024-01-01", periods=15, freq="B")
quotidien = pd.DataFrame(
    {
        "Open": np.arange(100.0, 115.0),
        "High": np.arange(100.0, 115.0) + 2,
        "Low": np.arange(100.0, 115.0) - 2,
        "Close": np.arange(100.0, 115.0) + 1,
        "Volume": np.full(15, 1000.0),
    },
    index=jours,
)
hebdo = S.to_weekly(quotidien)
verifier("agregation produit des semaines", len(hebdo) >= 2)
verifier("Open de la semaine = premier jour",
         np.isclose(hebdo["Open"].iloc[0], 100.0))
verifier("High de la semaine = max",
         np.isclose(hebdo["High"].iloc[0], 106.0))
verifier("Volume de la semaine = somme",
         np.isclose(hebdo["Volume"].iloc[0], 5000.0))
verifier("semaine en cours incomplete retiree",
         hebdo.index[-1] <= quotidien.index[-1])

jours_m = pd.date_range("2024-01-01", periods=95, freq="B")
quotidien_m = pd.DataFrame(
    {
        "Open": np.arange(100.0, 100.0 + len(jours_m)),
        "High": np.arange(100.0, 100.0 + len(jours_m)) + 2,
        "Low": np.arange(100.0, 100.0 + len(jours_m)) - 2,
        "Close": np.arange(100.0, 100.0 + len(jours_m)) + 1,
        "Volume": np.full(len(jours_m), 1000.0),
    },
    index=jours_m,
)
mensuel = S.to_monthly(quotidien_m)
verifier("agregation produit des mois", len(mensuel) >= 3)
verifier("Open du mois = premier jour ouvre",
         np.isclose(mensuel["Open"].iloc[0], 100.0))
verifier("mois en cours incomplet retire",
         mensuel.index[-1] <= quotidien_m.index[-1] + pd.Timedelta(days=31))

# Regression : le mois qui vient de se terminer ne doit pas etre supprime a
# tort quand le dernier jour CALENDAIRE du mois tombe un week-end/ferie.
def cadre_jours_ouvres(dernier_jour_ouvre, debut="2026-03-01"):
    jours = pd.bdate_range(debut, dernier_jour_ouvre)
    return pd.DataFrame({
        "Open": np.arange(100.0, 100.0 + len(jours)), "High": np.arange(100.0, 100.0 + len(jours)) + 2,
        "Low": np.arange(100.0, 100.0 + len(jours)) - 2, "Close": np.arange(100.0, 100.0 + len(jours)) + 1,
        "Volume": np.full(len(jours), 1000.0),
    }, index=jours)

# 31 mai 2026 = dimanche, derniere seance = vendredi 29 -> mai doit rester
cadre_mai = cadre_jours_ouvres("2026-05-29")
mensuel_mai = S.to_monthly(cadre_mai)
verifier(
    "mois termine un week-end : conserve (pas supprime a tort)",
    len(mensuel_mai) > 0 and mensuel_mai.index[-1] == pd.Timestamp("2026-05-31"),
)

# 31 aout 2026 = lundi (jour de bourse) : doit rester conserve (non-regression)
cadre_aout = cadre_jours_ouvres("2026-08-31")
mensuel_aout = S.to_monthly(cadre_aout)
verifier(
    "mois termine un jour de bourse : toujours conserve",
    len(mensuel_aout) > 0 and mensuel_aout.index[-1] == pd.Timestamp("2026-08-31"),
)

# Scan en plein milieu du mois : le mois en cours doit rester exclu
cadre_milieu = cadre_jours_ouvres("2026-06-15")
mensuel_milieu = S.to_monthly(cadre_milieu)
verifier(
    "mois reellement incomplet (milieu de mois) toujours exclu",
    len(mensuel_milieu) > 0 and mensuel_milieu.index[-1] == pd.Timestamp("2026-05-31"),
)

verifier("agreger_tf('D') = donnees inchangees", len(S.agreger_tf(quotidien, "D")) == len(quotidien))
verifier("agreger_tf('W') = to_weekly", len(S.agreger_tf(quotidien, "W")) == len(S.to_weekly(quotidien)))
verifier("agreger_tf('M') = to_monthly", len(S.agreger_tf(quotidien_m, "M")) == len(S.to_monthly(quotidien_m)))


# --- 6. Formatage du message -----------------------------------------------
print("\n7. Formatage du message")
signaux = [{
    "type": "stromboli", "ticker": "AAPL", "place": "US", "tf": "D", "sens": "haussier",
    "bougies": 4, "date": pd.Timestamp("2026-08-10"), "ha_close": 231.45,
    "ratio_corps": 0.021, "volume": 5e7, "volume_ratio": 1.8,
}]
message = S.formater(signaux, ["D", "W"])
verifier("ticker present", "AAPL" in message)
verifier("sens present", "HAUSSIER" in message)
verifier("volume present", "x1.8" in message)
verifier("message vide gere", "Aucun signal" in S.formater([], ["D"]))

signaux_fern = [{
    "type": "fernanda", "ticker": "MSFT", "place": "US", "tf": "D",
    "date": pd.Timestamp("2026-08-10"), "ha_close": 420.0,
    "stromboli_date": pd.Timestamp("2026-08-05"), "volume_ratio": 1.3,
}]
message_fern = S.formater(signaux_fern, ["D"])
verifier("fernanda : ticker present", "MSFT" in message_fern)
verifier("fernanda : libelle present", "FERNANDA" in message_fern)
verifier("fernanda : date stromboli reference", "05/08" in message_fern)

signaux_monthly = [{
    "type": "stromboli", "ticker": "BTCUSD", "place": "Crypto", "tf": "M", "sens": "haussier",
    "bougies": 4, "date": pd.Timestamp("2026-07-31"), "ha_close": 65000.0,
    "ratio_corps": 0.02,
}]
message_m = S.formater(signaux_monthly, ["D", "W", "M"], titre="STROMBOLI CRYPTO")
verifier("titre personnalise applique", "STROMBOLI CRYPTO" in message_m)
verifier("etiquette MONTHLY presente", "MONTHLY" in message_m)


# --- 8. Filet de secours Twelve Data -----------------------------------
print("\n8. Filet de secours Twelve Data")

verifier("mapping .PA -> XPAR", S._exchange_twelvedata("ALCPB.PA") == "XPAR")
verifier("mapping .AS -> XAMS", S._exchange_twelvedata("VPK.AS") == "XAMS")
verifier("mapping .BR -> XBRU", S._exchange_twelvedata("MELE.BR") == "XBRU")
verifier("ticker US -> pas d'exchange", S._exchange_twelvedata("AAPL") is None)

verifier(
    "desactive par defaut (cle vide)",
    S.TWELVEDATA_API_KEY == "" and S.telecharger_twelvedata(["ALCPB.PA"]) == {},
)

import unittest.mock as _mock

_reponse_ok = {
    "status": "ok",
    "values": [
        {"datetime": f"2026-{m:02d}-01", "open": "0.50", "high": "0.55", "low": "0.48", "close": "0.52", "volume": "125000"}
        for m in range(1, 13)
    ] * 3,
}


class _FausseReponseOk:
    def raise_for_status(self):
        pass

    def json(self):
        return _reponse_ok


S.TWELVEDATA_API_KEY = "cle_de_test"
with _mock.patch("requests.get", return_value=_FausseReponseOk()):
    _resultat = S.telecharger_twelvedata(["ALCPB.PA"])
verifier("parsing : ticker recupere", "ALCPB.PA" in _resultat)
verifier(
    "parsing : colonnes correctes",
    list(_resultat["ALCPB.PA"].columns) == ["Open", "High", "Low", "Close", "Volume"],
)
verifier("parsing : index trie chronologiquement", _resultat["ALCPB.PA"].index.is_monotonic_increasing)


class _FausseReponseErreur:
    def raise_for_status(self):
        pass

    def json(self):
        return {"status": "error", "code": 403, "message": "not available on your plan"}


with _mock.patch("requests.get", return_value=_FausseReponseErreur()):
    _resultat_erreur = S.telecharger_twelvedata(["ALCPB.PA"])
verifier("erreur de plan geree proprement", _resultat_erreur == {})

S.TWELVEDATA_API_KEY = ""  # restaure l'etat par defaut


# --- 9. Telechargement EODHD (source principale) ---------------------------
print("\n9. Telechargement EODHD")

verifier("mapping US ajoute .US", S._symbole_eodhd("AAPL") == "AAPL.US")
verifier("mapping .PA inchange", S._symbole_eodhd("ALCPB.PA") == "ALCPB.PA")
verifier("mapping .AS inchange", S._symbole_eodhd("AZRN.AS") == "AZRN.AS")
verifier("mapping .BR inchange", S._symbole_eodhd("MELE.BR") == "MELE.BR")
verifier("mapping indice .INDX inchange", S._symbole_eodhd("GSPC.INDX") == "GSPC.INDX")
verifier("mapping indice europeen .INDX inchange", S._symbole_eodhd("GDAXI.INDX") == "GDAXI.INDX")

verifier("periode 2y -> 730 jours", S._periode_en_jours("2y") == 730)
verifier("periode 1y -> 365 jours", S._periode_en_jours("1y") == 365)
verifier("periode 3mo -> 93 jours", S._periode_en_jours("3mo") == 93)

verifier(
    "desactive sans cle EODHD",
    S.EODHD_API_KEY == "" and S.telecharger(["AAPL"], "2y") == {},
)

_reponse_eodhd_ok = [
    {"date": f"2026-{m:02d}-01", "open": 100.0, "high": 105.0, "low": 98.0,
     "close": 103.0, "adjusted_close": 102.5, "volume": 125000}
    for m in range(1, 13)
] * 3


class _FausseReponseEodhdOk:
    def raise_for_status(self):
        pass

    def json(self):
        return _reponse_eodhd_ok


S.EODHD_API_KEY = "cle_de_test"
with _mock.patch("requests.get", return_value=_FausseReponseEodhdOk()):
    _resultat_eodhd = S.telecharger(["AAPL"], "2y")
verifier("EODHD : ticker recupere", "AAPL" in _resultat_eodhd)
verifier(
    "EODHD : colonnes correctes",
    list(_resultat_eodhd["AAPL"].columns) == ["Open", "High", "Low", "Close", "Volume"],
)
verifier(
    "EODHD : utilise adjusted_close",
    _resultat_eodhd["AAPL"]["Close"].iloc[0] == 102.5,
)
verifier("EODHD : index trie chronologiquement", _resultat_eodhd["AAPL"].index.is_monotonic_increasing)

S.EODHD_API_KEY = ""  # restaure l'etat par defaut


# --- 10. Fernando / Stromboli baissier (analyse uniquement) ----------------
print("\n10. Fernando / Stromboli baissier (analyse uniquement)")

_index_bas = pd.date_range("2024-01-01", periods=25, freq="D")
_lignes_bas = [[100.0, 101.0, 99.0, 100.0] for _ in range(9)]
_lignes_bas += [
    [100.0, 104.0, 100.0, 103.0],
    [103.0, 107.0, 103.0, 106.0],
    [106.0, 110.0, 106.0, 109.0],
]
_lignes_bas.append([109.0, 111.0, 107.0, 108.9])  # doji
for _k in range(12):
    _base = 108 - _k * 4
    _lignes_bas.append([_base, _base + 1, _base - 6, _base - 5])

_ha_bas = pd.DataFrame(
    _lignes_bas, columns=["open", "high", "low", "close"], index=_index_bas[: len(_lignes_bas)]
)
_ha_bas["volume"] = 1_000_000.0

_trouve_baissier = S.detecter_stromboli_baissier(_ha_bas, 12)
verifier("stromboli baissier detecte", _trouve_baissier is not None)
verifier("sens correctement etiquete baissier", _trouve_baissier and _trouve_baissier["sens"] == "baissier")

_occ_fernando = S.detecter_fernando_series(_ha_bas)
verifier("fernando detectee apres stromboli baissier", len(_occ_fernando) >= 1)
verifier(
    "fernando posterieure au doji (index 12)",
    _occ_fernando and _occ_fernando[0]["type"] == "fernando" and _occ_fernando[0]["index"] > 12,
)

# Inversion du P&L short : une baisse de prix doit ressortir en positif
_closes_test = [100.0] * 10
_closes_test[3] = 90.0  # -10% au bout de 3 jours


def _rendement_test(i, h, short):
    variation = (_closes_test[i + h] - _closes_test[i]) / _closes_test[i] * 100
    return -variation if short else variation


verifier("rendement long negatif sur une baisse", _rendement_test(0, 3, short=False) == -10.0)
verifier("rendement short positif sur la meme baisse", _rendement_test(0, 3, short=True) == 10.0)

# resume_backtest avec Fernando : verifie la section additionnelle
_df_fernando_test = pd.DataFrame([{
    "ticker": "TEST", "place": "US", "date": _ha_bas.index[14],
    "stromboli_date": _ha_bas.index[12], "prix_entree": 100.0,
    "rendement_1j": 2.0, "rendement_3j": 5.0, "rendement_5j": 8.0,
    "rendement_10j": None, "rendement_20j": None,
}])
_rapport_fernando = S.resume_backtest(
    pd.DataFrame(), pd.DataFrame(), annees=1, df_fernando=_df_fernando_test,
)
verifier("section Fernando presente dans le rapport", "FERNANDO" in _rapport_fernando)
verifier("mention P&L short dans le rapport", "vente a decouvert" in _rapport_fernando)

_rapport_sans_fernando = S.resume_backtest(pd.DataFrame(), pd.DataFrame(), annees=1)
verifier("pas de section Fernando si non demandee", "FERNANDO" not in _rapport_sans_fernando)




# --- 11. Backtest retour a la moyenne (RSI-2 / IBS) ------------------------
print("\n11. Backtest retour a la moyenne (RSI-2 / IBS)")

verifier("RSI-2 = 100 en hausse continue", S.calcul_rsi(np.array([100, 101, 102, 103, 104, 105.0]), 2)[-1] == 100.0)
verifier("RSI-2 = 0 en baisse continue", S.calcul_rsi(np.array([105, 104, 103, 102, 101, 100.0]), 2)[-1] == 0.0)


def _rsi_reference(c, p=14):
    s = pd.Series(c)
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1 / p, adjust=False).mean()
    al = l.ewm(alpha=1 / p, adjust=False).mean()
    return (100 - 100 / (1 + ag / al)).to_numpy()


_c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 400))
_ecart = np.nanmax(np.abs(S.calcul_rsi(_c, 14)[60:] - _rsi_reference(_c, 14)[60:]))
verifier("RSI conforme a une reference Wilder independante", _ecart < 0.5)

_cadre_ibs = pd.DataFrame({"High": [110.0, 110.0, 100.0], "Low": [100.0, 100.0, 100.0], "Close": [102.0, 110.0, 100.0]})
_ibs = S.calcul_ibs(_cadre_ibs)
verifier("IBS = 0.2 pour une cloture dans les 20% bas", abs(_ibs[0] - 0.2) < 1e-9)
verifier("IBS = 1.0 pour une cloture au plus haut", _ibs[1] == 1.0)
verifier("IBS = NaN si range nul", np.isnan(_ibs[2]))

_n = 320
_idx = pd.bdate_range("2024-01-01", periods=_n)
_base = np.linspace(100, 160, _n)
_base[250] -= 8
_base[251] -= 12
_cadre_rm = pd.DataFrame(
    {"Open": _base, "High": _base + 1, "Low": _base - 1, "Close": _base, "Volume": 1000.0}, index=_idx
)
_trades = S._trades_retour_moyenne(_cadre_rm, "rsi2")
verifier("un trade genere sur le repli dans la tendance haussiere", len(_trades) == 1)
verifier("entree le premier jour du repli (RSI-2 deja < 10)", _trades and _trades[0]["date_entree"] == _idx[250])
verifier("trade gagnant a la reprise", _trades and _trades[0]["rendement"] > 0)
verifier("sortie sur la MM5", _trades and _trades[0]["motif"] == "mm5")

# Stop loss : un crash brutal juste apres l'entree doit declencher le stop
# exactement au niveau du stop, pas a la cloture (plus negative) du jour.
_base_crash = _base.copy()
_base_crash[252] = _base_crash[251] * 0.80  # -20% le lendemain de l'entree
_cadre_crash = pd.DataFrame(
    {"Open": _base_crash, "High": _base_crash + 1, "Low": _base_crash - 1, "Close": _base_crash, "Volume": 1000.0},
    index=_idx,
)
_trades_stop = S._trades_retour_moyenne(_cadre_crash, "rsi2", stop_pct=-8.0)
verifier("stop loss declenche sur un crash brutal", _trades_stop and _trades_stop[0]["motif"] == "stop_loss")
verifier(
    "rendement au niveau exact du stop, pas la cloture reelle",
    _trades_stop and abs(_trades_stop[0]["rendement"] - (-8.0)) < 0.01,
)
verifier(
    "sans stop_pct, comportement inchange (retrocompatibilite)",
    S._trades_retour_moyenne(_cadre_rm, "rsi2", stop_pct=None) == _trades,
)

# Validation croisee : un renversement synthetique (1ere moitie 100% gagnante,
# 2e moitie 100% perdante) doit etre detecte distinctement dans les 2 blocs.
_trades_synthetiques = pd.DataFrame({
    "date_entree": pd.date_range("2024-01-01", periods=40, freq="15D"),
    "jours": [3] * 40,
    "rendement": [1.0] * 20 + [-1.0] * 20,
    "motif": ["mm5"] * 40,
    "place": ["US"] * 40,
})
_rapport_vc = S.resume_validation_croisee({"rsi2": _trades_synthetiques}, annees=2)
verifier("validation croisee : sections presentes", "DECOUVERTE" in _rapport_vc and "VALIDATION" in _rapport_vc)
verifier("validation croisee : decouverte a 100% de reussite", "reussite 100.0%" in _rapport_vc)
verifier("validation croisee : validation a 0% de reussite", "reussite   0.0%" in _rapport_vc)

_rapport_vc_vide = S.resume_validation_croisee({"rsi2": pd.DataFrame()}, annees=2)
verifier("validation croisee : echantillon vide gere", "echantillon trop petit" in _rapport_vc_vide)

_base_baisse = np.linspace(160, 100, _n)  # tendance baissiere : filtre MM200 doit bloquer
_cadre_baisse = pd.DataFrame(
    {"Open": _base_baisse, "High": _base_baisse + 1, "Low": _base_baisse - 1, "Close": _base_baisse, "Volume": 1000.0},
    index=_idx,
)
verifier("aucun trade sous la MM200 (filtre de tendance)", S._trades_retour_moyenne(_cadre_baisse, "rsi2") == [])

_rapport_rm = S.resume_retour_moyenne({"rsi2": pd.DataFrame(_trades).assign(place="US"), "ibs": pd.DataFrame()}, annees=2)
verifier("rapport RM : section RSI-2 presente", "RSI-2" in _rapport_rm)
verifier("rapport RM : section IBS vide geree", "aucun trade" in _rapport_rm)


print("\n" + "=" * 50)
if echecs:
    print(f"{len(echecs)} ECHEC(S) : {echecs}")
    raise SystemExit(1)
print("Tous les tests passent.")
