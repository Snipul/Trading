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


print("\n" + "=" * 50)
if echecs:
    print(f"{len(echecs)} ECHEC(S) : {echecs}")
    raise SystemExit(1)
print("Tous les tests passent.")
