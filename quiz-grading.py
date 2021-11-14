#!/usr/bin/env python3
# coding: utf-8

"""
Examples
````````
    ./quiz-grading.py \
        -i ../Quiz-3-2020-12-17/Quiz\ DataCamp\ 3.csv \
        -r ../Quiz-3-2020-12-17/Quiz\ DataCamp\ 3 - Ref.csv \
        -c 4 --anonym --to csv

    ./quiz-grading.py \
        -i "../Proba/Partiel/Partiel-2021 resultats.csv" \
        -r "../Proba/Partiel/Partiel-2021 resultats - Reference.csv" \
        -c 3 --anonym --to csv --separated-coef
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO: change with configuration file to give as input (columns namings)
LABELS = {"id": "Student ID", "first": "First name", "last": "Last name"}
# LABELS = {"id": "ID", "first": "First", "last": "Last"}


def lookup(df, row_labels, col_labels) -> np.ndarray:
    """
    Label-based "fancy indexing" function for DataFrame (simplified).
    Given equal-length arrays of row and column labels, return an
    array of the values corresponding to each (row, col) pair.

    Parameters
    ----------
    row_labels : Sequence
        The row labels to use for lookup.
    col_labels : Sequence
        The column labels to use for lookup.

    Returns
    -------
    numpy.ndarray
        The found values.
    """
    row_idx = df.index.get_indexer(row_labels)
    col_idx = df.columns.get_indexer(col_labels)
    flat_index = row_idx * len(df.columns) + col_idx
    return df.values.flat[flat_index]


def mapping(from_, to_):
    return {old: new for old, new in zip(from_, to_)}


def append_to_basename(fp: Path, appendix):
    return fp.with_name(f"{fp.stem}{appendix}").with_suffix(fp.suffix)


class Grader:
    def __init__(self, args):
        self.responses_raw_df, self.weights_df, self.reference_df = None, None, None
        self.results_df, self.grades_df = None, None
        self.headers = {}

        self.args = args
        self.resp_fp = Path(args.input)
        self.ref_fp = Path(args.ref)

        self.load_data()
        self.make_headers()
        self.cleanup_data()

        self.choose_output_headers()

    def load_data(self):
        self.responses_raw_df = pd.read_csv(self.resp_fp)

        self.weights_df = (
            pd.read_csv("weights.csv")
            .set_index("Coef")
            .rename(columns={"Right": True, "Wrong": False})
        )
        self.reference_df = pd.read_csv(self.ref_fp).set_index("Q_id")

    @staticmethod
    def rename_headers(basename, old_titles):
        return [f"{basename}{el}" for el in range(1, len(old_titles) // 2 + 1)]

    @staticmethod
    def import_range(column_limits):
        try:
            left, right, *_ = column_limits
        except ValueError:
            left = column_limits[0]
            right = None

        return left, right

    @staticmethod
    def alternate_list(l1, l2):
        return list(itertools.chain.from_iterable(zip(l1, l2)))

    def make_headers(self):
        headers = self.headers
        left, right = self.import_range(self.args.response_column_range)
        headers["resp_old_headers"] = self.responses_raw_df.columns[left:right]
        headers["questions"] = self.rename_headers("Q", headers["resp_old_headers"])
        headers["coefficients"] = [f"C{el[1:]}" for el in headers["questions"]]
        headers["responses"] = [f"R{el[1:]}" for el in headers["questions"]]
        headers["values"] = [f"V{el[1:]}" for el in headers["questions"]]

    def cleanup_data(self):
        if self.args.separated_coef is True:
            quest_coef = self.headers["questions"] + self.headers["coefficients"]
        else:
            quest_coef = self.alternate_list(
                self.headers["questions"], self.headers["coefficients"]
            )
        self.responses_df = self.responses_raw_df.rename(
            columns=mapping(self.headers["resp_old_headers"], to_=quest_coef)
        )
        # Sanity check to verify if response columns were well selected
        assert (
            self.responses_df[self.headers["coefficients"]].dtypes == np.dtype("int")
        ).all()

    def choose_output_headers(self):
        headers = self.headers

        headers["grades"] = self.alternate_list(headers["responses"], headers["values"])
        if self.args.adjust:
            headers["totals"] = ["Total", "Total_neg", "Baseline", "Total_adjusted"]
        else:
            headers["totals"] = ["Total", "Total_neg"]

        if self.args.anonym:
            headers["ids"] = [LABELS["id"]]
        else:
            headers["ids"] = [LABELS["last"], LABELS["first"], LABELS["id"]]

        headers["all"] = headers["ids"] + headers["grades"] + headers["totals"]

    def assess_responses(self):
        headers = self.headers
        questions_df = self.responses_df[headers["questions"]]

        self.matches_df = questions_df.eq(self.reference_df.Response)
        self.matches_df.rename(
            columns=mapping(headers["questions"], to_=headers["responses"]),
            inplace=True,
        )

    def grade(self):
        headers = self.headers

        values = lookup(
            self.weights_df,
            self.responses_df[headers["coefficients"]].values.flatten(),
            self.matches_df.values.flatten(),
        )

        self.grades_df = pd.DataFrame(
            values.reshape(self.matches_df.shape), columns=headers["values"]
        )

    def compute_results(self):
        grades_df, matches_df = self.grades_df, self.matches_df

        results_df = self.responses_df.join(matches_df * 1).join(
            grades_df.astype("int")
        )
        results_df["Total"] = grades_df.mean(axis="columns").clip(0, 20).round(1)
        results_df["Baseline"] = (matches_df * 18).mean(axis="columns").round(1)
        results_df["Total_neg"] = (
            results_df[self.headers["values"]]
            .clip(lower=None, upper=0)
            .sum(axis="columns")
            .astype("int")
        )
        results_df["Total_adjusted"] = results_df["Total"].clip(results_df["Baseline"])

        self.results_df = results_df

    def process(self):
        self.assess_responses()
        self.grade()
        self.compute_results()

    def save(self, to_):
        if to_ == "csv":
            self.results_df[self.headers["all"]].to_csv(
                append_to_basename(self.resp_fp, " - Results"), float_format="%.1f"
            )

    def print(self):
        headers = self.headers

        # To avoid printing ID as float (workaround for .to_markdown() issues)
        results_df = self.results_df.copy()
        results_df[f"{LABELS['id']} str"] = results_df[LABELS["id"]].astype(str)

        print()
        print(results_df[headers["all"]].to_markdown(index=False))
        print()
        print(results_df[headers["grades"]].describe().round(2).to_markdown())
        print()
        print(results_df[headers["totals"]].describe().round(2).to_markdown())
        print()

    def plot_hist(self):
        self.results_df.Total_adjusted.hist()
        plt.show()


def argp():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input csv file to process."
    )
    parser.add_argument("-r", "--ref", required=True, help="Input csv file to process.")
    parser.add_argument("--to", choices=["csv"], help="Export results to csv file.")
    parser.add_argument(
        "-c",
        "--response-column-range",
        nargs="+",
        type=int,
        default=[4],
        help="Response column range (first or [first, last]).",
    )
    parser.add_argument(
        "--separated-coef",
        action="store_true",
        help="Separated certitude coefficients (default: alternating).",
    )
    parser.add_argument(
        "--adjust", action="store_true", help="Export adjusted to baseline total."
    )
    parser.add_argument("--anonym", action="store_true", help="Remove student names.")
    parser.add_argument("--plot-hist", action="store_true", help="Plot histogram.")

    return parser.parse_args()


def main(args):

    grader = Grader(args)
    grader.process()
    grader.print()

    if args.to is not None:
        grader.save(args.to)

    if args.plot_hist:
        grader.plot_hist()


if __name__ == "__main__":
    try:
        main(argp())
    except KeyboardInterrupt:
        print("\nInterrupted\n")
