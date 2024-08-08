# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "typer",
# ]
# ///

import pathlib

import pandas
import typer


def main(datafile: pathlib.Path, index_col: str):
    f = pandas.read_csv(datafile)
    print("\n".join(f[index_col].values[1:5]))


if __name__ == "__main__":
    typer.run(main)
