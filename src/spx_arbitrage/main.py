"""
main.py

Interactive CLI entry point for the project (run with `spx-arb`).

Walks the user through both parts of the project (Part 1: arbitrage check,
Part 2: optimal exit portfolio), handles all communication between the
modules, and writes the result to a csv file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import spx_arbitrage.data as data
import spx_arbitrage.solver as solver

DEFAULT_OUTPUT_DIR = Path("output")
POSITION_LIMIT = 100.0

# CBOE files start with a blank row, a last S&P 500 price row, and a header row
CBOE_HEADER_ROWS = 3


def clean_path_input(text: str) -> Path:
    """
    Strip whitespace and surrounding quotes, expand.
    """
    return Path(text.strip().strip("\"'")).expanduser()


def prompt_option_file() -> tuple[Path, data.OptionFile]:
    """
    Ask for the input csv until it can be loaded. Returns the path and the OptionFile (defined in data.py).
    """
    while True:
        path = clean_path_input(input("Specify data input path (csv): "))
        try:
            option_file = data.load_file_as_optionfile(path)
        except Exception as e:
            print(f"Error reading data file ({e}). Please re-enter the path to the csv file.")
            continue
        return path, option_file


def prompt_output_location() -> Path:
    """
    Ask where to save results. Returns either a directory or a full file path.

    Valid inputs:
    - Enter (blank): the default output directory.
    - A path ending in .csv: used as the full output file path.
    - Anything else: treated as a directory.
    """
    while True:
        text = input(
            "Output directory or full file path ending in .csv "
            f"[Enter for '{DEFAULT_OUTPUT_DIR}/']: "
        )
        location = clean_path_input(text) if text.strip() else DEFAULT_OUTPUT_DIR
        folder = location.parent if location.suffix.lower() == ".csv" else location
        try:
            folder.mkdir(parents=True, exist_ok=True)  # try to make folder (+parents), no error if it already exists
        except Exception as e:
            print(f"Cannot use that location ({e}). Please try again.")
            continue
        return location


def default_filename(data_path: Path, *tags: str) -> str:
    """
    Build a default filename from the input file's name plus optional tags
    (e.g. part and expiry), with whitespace replaced by underscores.
    """
    name = "_".join([data_path.stem, *tags])
    return "_".join(name.split()) + ".csv"


def resolve_output_path(location: Path, data_path: Path, *tags: str) -> Path:
    """
    Convert the user's output path choice into a concrete file path.
    If file name not given, names csv using information about query.
    """
    if location.suffix.lower() == ".csv":
        return location
    return location / default_filename(data_path, *tags)


def prompt_problem_type() -> str:
    """
    Ask whether solving arbitrage detection or exit portfolio construction.
    """

    while True:
        problem = input("Enter 'a' for arbitrage detection or 'c' for exit portfolio construction: ").strip().lower()

        if problem in {"a", "c"}:
            return problem

        print("Invalid choice. Please enter 'a' or 'c'.")


def prompt_float(message: str) -> float:
    """
    Ask until the input parses as a float.
    """

    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Please enter a number.")


def parse_is_call(value) -> bool:
    """
    Accepts True/False, 1/0, C/P, call/put.
    """

    text = str(value).strip().lower()
    if text in {"true", "1", "c", "call"}:
        return True
    if text in {"false", "0", "p", "put"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as call/put")


def load_holdings(path: Path, chain) -> np.ndarray:
    """
    Read a holdings csv and match it to the chain.

    Expected columns: strike, is_call, quantity
    (positive quantity is long, negative quantity is short).
    A contract in the file that is not in the chain is an error. 
    Duplicate rows are summed.
    """

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"strike", "is_call", "quantity"} - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s): {', '.join(sorted(missing))}")

    index = {(float(k), bool(c)): i for i, (k, c) in enumerate(zip(chain.strikes, chain.is_call))}

    holdings = np.zeros(len(index))
    for strike, is_call, quantity in zip(df["strike"], df["is_call"], df["quantity"]):
        key = (float(strike), parse_is_call(is_call))
        if key not in index:
            kind = "call" if key[1] else "put"
            raise ValueError(f"{kind} with strike {key[0]:g} is not in the chain for this expiry")
        holdings[index[key]] += float(quantity)

    return holdings


def prompt_holdings(chain) -> np.ndarray:
    """
    Ask whether the position is a single option or multiple.

    - Single: prompt for strike, call/put and quantity.
    - Multiple: prompt for a holdings csv (columns: strike, is_call, quantity).

    Returns a holdings vector matched to the chain.
    """
    
    while True:
        choice = input("Do you hold only a single option position? (y/n): ").strip().lower()
        if choice in {"y", "n"}:
            break
        print("Invalid choice. Please enter 'y' or 'n'.")

    if choice == "n":
        while True:
            path = clean_path_input(input("Specify holdings csv (columns: strike, is_call, quantity): "))
            try:
                return load_holdings(path, chain)
            except Exception as e:
                print(f"Error reading holdings file ({e}). Please try again.")

    index = {
        (float(k), bool(c)): i
        for i, (k, c) in enumerate(zip(chain.strikes, chain.is_call))
    }

    while True:
        strike = prompt_float("Strike: ")
        try:
            is_call = parse_is_call(input("Call or put (c/p): "))
        except ValueError as e:
            print(f"{e}. Please enter 'c' or 'p'.")
            continue

        key = (strike, is_call)
        if key in index:
            break
        kind = "call" if is_call else "put"
        print(f"No {kind} with strike {strike:g} in the chain for this expiry. Please try again.")

    quantity = prompt_float("Quantity (positive = long, negative = short): ")

    holdings = np.zeros(len(chain.strikes))
    holdings[index[key]] = quantity
    return holdings


def build_results(chain, x: np.ndarray, holdings: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build a table from a solution's decision vector.
    """
    
    x = np.asarray(x, dtype=float).ravel()
    x = np.round(x, 6) + 0.0  # round off solver noise; + 0.0 turns -0.0 into 0.0
    n = len(chain.strikes)
    n = len(chain.strikes)

    long, short = x[:n], x[n:2 * n]
    results = pd.DataFrame({
        "instrument": np.where(np.asarray(chain.is_call, dtype=bool), "call", "put"),
        "strike": np.asarray(chain.strikes, dtype=float),
        "long": long,
        "short": short,
        "net": long - short,
    })

    if holdings is not None:
        results.insert(2, "current_holdings", holdings)
        bond_long, bond_short = x[2 * n:]
        bond_row = pd.DataFrame([{
            "instrument": "bond",
            "long": bond_long,
            "short": bond_short,
            "net": bond_long - bond_short,
        }])
        results = pd.concat([results, bond_row], ignore_index=True)

    return results


def main() -> None:
    data_path, option_file = prompt_option_file()
    output_location = prompt_output_location()

    expiry = data.prompt_expiry(option_file.data)
    series = data.prompt_series(option_file.data, expiry)
    chain = data.build_chain(option_file, expiry, series)

    problem = prompt_problem_type()
    tag = "arbitrage" if problem == "a" else "exit"

    output_path = resolve_output_path(output_location, data_path, tag, expiry.date().isoformat(), series)
    print(f"Results will be saved to: {output_path}")

    if problem == "a":
        lp = solver.Solver(
            strikes=chain.strikes,
            is_call=chain.is_call,
            bid=chain.bid,
            ask=chain.ask,
            position_limit=POSITION_LIMIT,
        )
        try:
            z, x = lp.main()
        except RuntimeError as e:
            print('Solving error:', e)
            return

        if z < 0:
            print(f"Arbitrage detected! Profit: {-z}")
            build_results(chain, x).to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
        else:
            print("No arbitrage detected. No file written.")
        return

    else:
        holdings = prompt_holdings(chain)
        rate = prompt_float("Risk-free rate: ")
        time_to_expiry = prompt_float("Time to expiry (years): ")

        lp = solver.Solver(
            strikes=chain.strikes,
            is_call=chain.is_call,
            bid=chain.bid,
            ask=chain.ask,
            position_limit=POSITION_LIMIT,
            holdings=holdings,
            rate=rate,
            time_to_expiry=time_to_expiry,
        )
        try:
            z, x = lp.main()
        except RuntimeError as e:
            print('Error when solving:', e)
            return

        print(f"Exit cost: {z}")

        results = build_results(chain, x, holdings)
        bond = results.iloc[-1]
        print(f"Bond position: {bond['long']:.4f} long, {bond['short']:.4f} short (net {bond['net']:.4f})")

        results.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()