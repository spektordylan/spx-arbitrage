"""
data.py

Handles reading and cleaning CBOE data. 
Prompts user for expiry and series, all other prompting handled by main.py
"""

from dataclasses import dataclass
from pathlib import Path
import csv
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 31536000

NUM_HEADER_ROWS = 3

SERIES_INFO = {"SPX": "09:30", "SPXW": "16:00"} # series names and settlement times

EASTERN = ZoneInfo("America/New_York")

# select necessary columns, rename with unique names
COLUMN_POSITIONS = {
    0: "expiry",
    1: "call_ticker",
    4: "call_bid",
    5: "call_ask",
    11: "strike",
    12: "put_ticker",
    15: "put_bid",
    16: "put_ask"
}

@dataclass
class OptionFile:
    """Cleaned CBOE csv file."""
    spot: float # "Last: " price in second header row
    quote_time: pd.Timestamp # read from "Date: " in third header row
    data: pd.DataFrame # actual data, columns are all names in COLUMN_POSITIONS + a "series" column + an "expiry column" with pd.Timestamp

@dataclass
class OptionChain:
    """
    Option chain to be formed from data.
    One expiry, one series (weekly/monthly).
    """
    spot: float
    quote_time: pd.Timestamp
    expiry_time: pd.Timestamp
    series: str 
    strikes: np.ndarray
    call_bid: np.ndarray
    call_ask: np.ndarray
    put_bid: np.ndarray
    put_ask: np.ndarray

    def settlement_time(self) -> pd.Timestamp:
        """
        Exact settlement time in US Eastern time.
        Monthly series are AM-settled, weekly series are PM-settled.
        """
        date_str = self.expiry_time.strftime("%Y-%m-%d")
        time_str = SERIES_INFO[self.series]
        return pd.Timestamp(f"{date_str} {time_str}", tz=EASTERN)

    def time_to_expiry(self) -> float:
        """Time between quote_time and expiry for pricing zero-coupon bond (user-supplied continously compounding rate)."""
        return (self.settlement_time() - self.quote_time).total_seconds() / SECONDS_PER_YEAR # a little more precision from seconds


def read_header_rows(path: Path) -> tuple[float, pd.Timestamp]:
    """
    Read header rows above the data, return (spot, quote_time).
    quote_time is assumed to be in US Eastern time.
    """
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r][:2]   # skip blank lines, keep 2 header rows

    # ['S&P 500 INDEX', 'Last: 4774.2402', 'Change:  -0.5098']
    spot = float(rows[0][1].split(":")[1])

    # ['Date: December 27, 2023 at 2:36 PM EST', 'Bid: ...', ...]
    text = rows[1][0].removeprefix("Date:").strip()      # 'December 27, 2023 at 2:36 PM EST'
    text = text.rsplit(" ", 1)[0].replace(" at ", " ")   # drop 'EST', then the 'at'
    quote_time = pd.Timestamp(text).tz_localize(EASTERN) # parses '... 2:36 PM'
    return spot, quote_time


def load_file_as_optionfile(path: Path) -> OptionFile:
    """
    Load CBOE csv as an OptionFile object.
    """
    spot, quote_time = read_header_rows(path)

    # skip the blank line, the two header rows, and the column-name row
    # header=None because the file's column names are not unique
    df = pd.read_csv(
        path,
        skiprows=NUM_HEADER_ROWS + 1,
        header=None,
        usecols=list(COLUMN_POSITIONS),
        names=list(COLUMN_POSITIONS.values()),
    )

    df["expiry"] = pd.to_datetime(df["expiry"], format="%a %b %d %Y")
    for col in ("strike", "call_bid", "call_ask", "put_bid", "put_ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["expiry", "strike"])

    # Determine series from alphabetic prefix
    df["series"] = df["call_ticker"].str.extract(r"^([A-Z]+)")[0]

    df = df.sort_values(["expiry", "series", "strike"]).reset_index(drop=True)
    return OptionFile(spot=spot, quote_time=quote_time, data=df)


def _clean_quotes(bid: np.ndarray, ask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cleans bid/ask quotes to handle market implications of missing/zero prices.
    """
    bid = np.where(np.isnan(bid) | (bid < 0), 0.0, bid) # NaN/nonpositive bid means no buyers => 0.0 price
    bad = np.isnan(ask) | (ask <= 0) | (bid > ask)      # get all bad quotes (missing/nonpositive ask, bid crossed ask)
    return np.where(bad, 0.0, bid), np.where(bad, np.nan, ask)  # bad bid quotes become 0.0, bad asks become NaN (0.0 implies free option!)


def prompt_expiry(df: pd.DataFrame) -> pd.Timestamp:
    """
    Ask for an expiry until it matches at least one row.
    """
    expiries = df["expiry"].dt.normalize()
    available = ", ".join(d.date().isoformat() for d in sorted(expiries.drop_duplicates()))
    print(f"Available expiries: {available}")

    valid_expiry_given = False
    while not valid_expiry_given:
        text = input("Expiry (YYYY-MM-DD): ").strip()
        try:
            expiry = pd.Timestamp(text).normalize()
        except (ValueError, TypeError):
            print("Could not parse that date. Please try again.")
            continue
        if (expiries == expiry).any():
            valid_expiry_given = True
        else:
            print(f"No data for {expiry.date()}. Please choose from the list above.")

    return expiry


def prompt_series(df: pd.DataFrame, expiry: pd.Timestamp) -> str:
    """
    Return the series for a given expiry, asking only if both SPX and SPXW exist.
    """
    present = set(df.loc[df["expiry"].dt.normalize() == expiry, "series"])
    series_choices = [s for s in SERIES_INFO if s in present]  # order: SPX, SPXW

    if not series_choices:
        raise ValueError(f"no SPX or SPXW rows for {expiry.date()}; found: {sorted(present)}")
    if len(series_choices) == 1:
        print(f"Using {series_choices[0]}, the only series for {expiry.date()}.")
        return series_choices[0]

    valid_series_given = False
    while not valid_series_given:
        series = input("Series (SPX / SPXW): ").strip().upper()
        if series in SERIES_INFO:
            valid_series_given = True
        else:
            print("Please enter SPX or SPXW.")

    return series


def build_chain(option_file: OptionFile, expiry: str | pd.Timestamp, series: str) -> OptionChain:
    """
    Build an option chain with a single expiry and series out of an OptionFile.
    """
    expiry = pd.Timestamp(expiry).normalize()
    df = option_file.data
    sub = df[(df["expiry"] == expiry) & (df["series"] == series)]
    if sub.empty:
        available = sorted({(e.date().isoformat(), s) for e, s in zip(df["expiry"], df["series"])})
        raise ValueError(f"no rows for {expiry.date()} {series}; available: {available}")
    if sub["strike"].duplicated().any():
        raise ValueError("duplicate strikes in chain")
    sub = sub.sort_values("strike")

    call_bid, call_ask = _clean_quotes(sub["call_bid"].to_numpy(float), sub["call_ask"].to_numpy(float))
    put_bid, put_ask = _clean_quotes(sub["put_bid"].to_numpy(float), sub["put_ask"].to_numpy(float))

    chain = OptionChain(
        spot=option_file.spot,
        quote_time=option_file.quote_time,
        expiry_time=expiry,
        series=series,
        strikes=sub["strike"].to_numpy(float),
        call_bid=call_bid, call_ask=call_ask,
        put_bid=put_bid, put_ask=put_ask,
    )
    if chain.time_to_expiry() <= 0:
        raise ValueError("settlement is not after quote time")
    return chain