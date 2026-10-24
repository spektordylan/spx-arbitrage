"""
main.py

Interactive CLI entry point for the project (run with `spx-arb`).

Walks the user through both parts of the project (Part 1: arbitrage check,
Part 2: optimal exit portfolio), handles all communication between the
modules, and writes the result to a csv file.
"""

from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path("output")

# CBOE files start with a blank row, a last S&P 500 price row, and a header row
CBOE_HEADER_ROWS = 3

def clean_path_input(text: str) -> Path:
    """Strip whitespace and surrounding quotes, expand."""
    return Path(text.strip().strip("\"'")).userexpand()


def prompt_data_path() -> tuple[Path, pd.DataFrame]:
    """Ask for the input csv until it can be read. Returns the path and the raw data."""
    while True:
        path = clean_path_input(input("Specify data input path (csv): "))
        try:
            data = pd.read_csv(path, skiprows=CBOE_HEADER_ROWS, header=None)
        except Exception as e:
            print(f"Error reading data file ({e}). Please re-enter the path to the csv file.")
            continue
        return path, data


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
            folder.mkdir(parents=True, exist_ok=True) # try to make folder (+parents), no error if it already exists
        except Exception as e:
            print(f"Cannot use that location ({e}). Please try again.")
            continue
        return location


def default_filename(data_path: Path, *tags: str) -> str:
    """Build a default filename from the input file's name plus optional tags
    (e.g. part and expiry), with whitespace replaced by underscores."""
    name = "_".join([data_path.stem, *tags])
    return "_".join(name.split()) + ".csv"


def resolve_output_path(location: Path, data_path: Path, *tags: str) -> Path:
    """Convert the user's output path choice into a concrete file path."""
    if location.suffix.lower() == ".csv":
        return location
    return location / default_filename(data_path, *tags)


def main() -> None:
    data_path, option_data = prompt_data_path()
    output_location = prompt_output_location()

    # TODO: expiry -> series (only if both exist) -> Part 1 or Part 2
    # -> Part 2 additional inputs. Once part and expiry are known, pass them as tags,
    # e.g. resolve_output_path(output_location, data_path, "arb", "2022-10-21").
    output_path = resolve_output_path(output_location, data_path)
    print(f"Results will be saved to: {output_path}")

if __name__ == "__main__":
    main()