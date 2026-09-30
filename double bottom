"""
Détecteur de double bottom (v1) — EODHD (actions) + Kraken (crypto).

Usage :
  export EODHD_API_TOKEN=xxxx
  python double_bottom.py --symbol AAPL.US --source eodhd
  python double_bottom.py --symbol XBTUSD --source kraken
  python double_bottom.py --symbol AAPL.US --source eodhd --last-only --telegram

Règles (toutes paramétrables en haut du fichier) :
  - pivots par ZigZag (retournement >= ZIGZAG_PCT)
  - séquence Creux1 - Sommet - Creux2
  - Creux1 / Creux2 à moins de LOW_TOL d'écart
  - MIN_BARS..MAX_BARS bougies entre les deux creux
  - sommet intermédiaire >= MIN_RISE au-dessus des creux
  - signal = 1re clôture au-dessus du sommet (neckline), dans les BREAKOUT_WINDOW
    bougies suivant le 2e creux, sans casser les creux avant
"""
import argparse
import os
from dataclasses import dataclass

import pandas as pd
import requests

# ---------- Paramètres ----------
ZIGZAG_PCT = 0.05        # retournement minimal pour valider un pivot
LOW_TOL = 0.03           # écart max entre les 2 creux
MIN_BARS, MAX_BARS = 15, 60
MIN_RISE = 0.10          # sommet intermédiaire vs creux
BREAKOUT_WINDOW = 30     # bougies max après le 2e creux pour casser la neckline
VOL_MA = 20              # moyenne de volume
TREND_MA = 200           # filtre de tendance de fond (informatif)


# ---------- Données ----------
def fetch_eodhd(symbol: str, days: int = 900) -> pd.DataFrame:
    token = os.environ["EODHD_API_TOKEN"]
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    r = requests.get(
        f"https://eodhd.com/api/eod/{symbol}",
        params={"api_token": token, "fmt": "json", "period": "d", "from": start},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # ajustement cohérent de l'OHLC (splits/dividendes)
    k = df["adjusted_close"] / df["close"]
    for c in ("open", "high", "low"):
        df[c] = df[c] * k
    df["close"] = df["adjusted_close"]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_kraken(pair: str, interval: int = 1440) -> pd.DataFrame:
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": interval},
        timeout=30,
    )
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(js["error"])
    key = next(k for k in js["result"] if k != "last")
    df = pd.DataFrame(
        js["result"][key],
        columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
    )
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("date").astype(float)
    # on retire la dernière bougie (encore en cours)
    return df[["open", "high", "low", "close", "volume"]].iloc[:-1]


# ---------- Pivots ----------
def zigzag(df: pd.DataFrame, pct: float):
    """Retourne une liste [(index_position, 'L'|'H', prix)]."""
    hi, lo = df["high"].values, df["low"].values
    n = len(df)
    pivots = []
    direction = 0
    hi_i = lo_i = 0
    for i in range(1, n):
        if direction == 0:
            if hi[i] > hi[hi_i]:
                hi_i = i
            if lo[i] < lo[lo_i]:
                lo_i = i
            if hi[i] >= lo[lo_i] * (1 + pct) and lo_i < i:
                pivots.append((lo_i, "L", lo[lo_i]))
                direction, hi_i = 1, i
            elif lo[i] <= hi[hi_i] * (1 - pct) and hi_i < i:
                pivots.append((hi_i, "H", hi[hi_i]))
                direction, lo_i = -1, i
        elif direction == 1:
            if hi[i] > hi[hi_i]:
                hi_i = i
            elif lo[i] <= hi[hi_i] * (1 - pct):
                pivots.append((hi_i, "H", hi[hi_i]))
                direction, lo_i = -1, i
        else:
            if lo[i] < lo[lo_i]:
                lo_i = i
            elif hi[i] >= lo[lo_i] * (1 + pct):
                pivots.append((lo_i, "L", lo[lo_i]))
                direction, hi_i = 1, i
    return pivots


# ---------- Détection ----------
@dataclass
class DoubleBottom:
    i_low1: int
    i_peak: int
    i_low2: int
    i_break: int
    neckline: float
    low: float
    target: float
    vol_ok: bool
    trend_ok: bool


def detect_double_bottoms(df: pd.DataFrame) -> list[DoubleBottom]:
    piv = zigzag(df, ZIGZAG_PCT)
    close, low = df["close"].values, df["low"].values
    vol_ma = df["volume"].rolling(VOL_MA).mean().values
    trend_ma = df["close"].rolling(TREND_MA).mean().values
    found = []

    for k in range(len(piv) - 2):
        (i1, t1, p1), (ih, t2, ph), (i2, t3, p2) = piv[k], piv[k + 1], piv[k + 2]
        if (t1, t2, t3) != ("L", "H", "L"):
            continue
        if abs(p2 - p1) / p1 > LOW_TOL:
            continue
        if not (MIN_BARS <= i2 - i1 <= MAX_BARS):
            continue
        low_min = min(p1, p2)
        if ph / low_min - 1 < MIN_RISE:
            continue

        # recherche de la cassure de la neckline
        for j in range(i2 + 1, min(i2 + 1 + BREAKOUT_WINDOW, len(df))):
            if low[j] < low_min * (1 - LOW_TOL):  # figure invalidée
                break
            if close[j] > ph:
                vol_ok = bool(df["volume"].iloc[j] > vol_ma[j]) if not pd.isna(vol_ma[j]) else False
                trend_ok = bool(close[j] > trend_ma[j]) if not pd.isna(trend_ma[j]) else False
                found.append(
                    DoubleBottom(i1, ih, i2, j, ph, low_min, ph + (ph - low_min), vol_ok, trend_ok)
                )
                break
    return found


# ---------- Sortie ----------
def plot(df: pd.DataFrame, pats: list[DoubleBottom], symbol: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df.index, df["close"], lw=1, color="#333")
    for p in pats:
        ax.plot(df.index[[p.i_low1, p.i_low2]], [df["low"].iloc[p.i_low1], df["low"].iloc[p.i_low2]],
                "go", ms=7)
        ax.plot(df.index[p.i_peak], p.neckline, "rv", ms=7)
        ax.hlines(p.neckline, df.index[p.i_peak], df.index[p.i_break], colors="r", linestyles="--", lw=1)
        ax.plot(df.index[p.i_break], df["close"].iloc[p.i_break], "b^", ms=10)
    ax.set_title(f"{symbol} — double bottoms détectés ({len(pats)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def send_telegram(text: str):
    token, chat = os.environ["TELEGRAM_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text},
        timeout=15,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--source", choices=["eodhd", "kraken"], required=True)
    ap.add_argument("--last-only", action="store_true", help="ne garder que les cassures de la dernière bougie")
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = fetch_eodhd(a.symbol) if a.source == "eodhd" else fetch_kraken(a.symbol)
    pats = detect_double_bottoms(df)
    if a.last_only:
        pats = [p for p in pats if p.i_break == len(df) - 1]

    print(f"{a.symbol}: {len(df)} bougies, {len(pats)} double bottom(s)")
    for p in pats:
        print(
            f"  creux {df.index[p.i_low1].date()} / {df.index[p.i_low2].date()} | "
            f"neckline {p.neckline:.2f} | cassure {df.index[p.i_break].date()} | "
            f"objectif {p.target:.2f} | stop {p.low:.2f} | "
            f"volume>MA{VOL_MA}: {p.vol_ok} | >MM{TREND_MA}: {p.trend_ok}"
        )
        if a.telegram:
            send_telegram(
                f"📈 Double bottom {a.symbol}\nCassure {p.neckline:.2f} ({df.index[p.i_break].date()})\n"
                f"Objectif {p.target:.2f} | Stop {p.low:.2f}\nVol OK: {p.vol_ok} | Tendance OK: {p.trend_ok}"
            )

    out = a.out or f"double_bottom_{a.symbol.replace('.', '_')}.png"
    plot(df, pats, a.symbol, out)
    print(f"Graphique : {out}")


if __name__ == "__main__":
    main()
