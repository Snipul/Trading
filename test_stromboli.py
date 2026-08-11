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
verifier("3 vertes + doji -> stromboli", resultat is not None)
verifier("sens baissier", resultat and resultat["sens"] == "baissier")

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


# --- 4. Agregation hebdomadaire --------------------------------------------
print("\n4. Agregation hebdomadaire")
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


# --- 5. Formatage du message -----------------------------------------------
print("\n5. Formatage du message")
signaux = [{
    "ticker": "AAPL", "place": "US", "tf": "D", "sens": "haussier",
    "bougies": 4, "date": pd.Timestamp("2026-08-10"), "ha_close": 231.45,
    "ratio_corps": 0.021, "volume": 5e7, "volume_ratio": 1.8,
}]
message = S.formater(signaux, ["D", "W"])
verifier("ticker present", "AAPL" in message)
verifier("sens present", "HAUSSIER" in message)
verifier("volume present", "x1.8" in message)
verifier("message vide gere", "Aucun signal" in S.formater([], ["D"]))


print("\n" + "=" * 50)
if echecs:
    print(f"{len(echecs)} ECHEC(S) : {echecs}")
    raise SystemExit(1)
print("Tous les tests passent.")
