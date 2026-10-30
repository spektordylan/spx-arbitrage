# spx-arbitrage

Linear programming methods for detecting static arbitrage and optimizing position liquidation in S&P 500 options markets.

This is a project for APPM 4720 - Open Topics in Applied Mathematics: Mathematical Finance, instructed by Dr. Daniel Brown.

## Build and Run

### Install uv

This project is managed with [uv](https://docs.astral.sh/uv/). If you don't have it, install it with one of the following.

macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Or, if you already have Python and pip:

```bash
pip install uv
```

Restart your terminal, then check that it worked:

```bash
uv --version
```

You do not need to install Python separately. The project requires Python 3.14, and uv will download it automatically the first time you run `uv sync`.

### Run

From the repository root:

```bash
uv sync
uv run python -m spx_arbitrage.main
```

The program is interactive and asks for everything it needs:

1. The path to a CBOE option chain csv (put these in `data/`).
2. Where to save results (press Enter for `output/`).
3. An expiry and, if both exist for that expiry, a series (SPX or SPXW).
4. Which problem to solve: arbitrage detection (`a`) or exit portfolio construction (`c`).

For exit portfolio construction it also asks for your current position and the risk-free rate (continuously compounded). A single option position is entered directly at the prompt. Multiple positions are read from a csv with the columns:

```csv
strike,is_call,quantity
4500,C,10
4600,P,-5
```

A positive quantity is long and a negative quantity is short.

Results are saved as a csv in the output location.

## Troubleshooting

- **`uv: command not found`**: restart your terminal after installing uv so it picks up the new PATH. If it still isn't found, follow the instructions printed at the end of the installer.
- **Error reading data file**: the file must be a CBOE option chain csv with the original header rows (a blank row, the last S&P 500 price row, and a date row) before the column names.
- **Error reading holdings file**: check that the columns are named `strike`, `is_call` and `quantity`, and that every strike and call/put combination exists in the chain for the chosen expiry.
- **LP solver failed**: the problem was infeasible or numerically unstable. For exit portfolios, this can mean the position cannot be offset because the options needed have no ask price.

## AI Acknowledgement

GPT-6 was used for:

- Initially determining the optimal project build
- Writing new `prompt_for_[]` functions after the first few were written manually
- Ensuring my translation of the LP from paper to code was correct
- Convoluted pandas date handling/timezone logic
- Type annotations, formatting, comments, numpy abuse, "why doesn't this work"...
- Highly accelerated last-minute debugging (pls don't look at commit history)
- This README