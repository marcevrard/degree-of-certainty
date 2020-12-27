#!/usr/bin/env python3
# coding: utf-8

"""
Examples
````````
./quiz-grading.py -i ../DataCamp/Quiz-3-2020-12-17/Quiz\ DataCamp\ 3.csv -r 4
    --anonym --to csv
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rename_headers(basename, old_titles):
    return [f"{basename}{el}" for el in range(1, len(old_titles) // 2 + 1)]


def make_headers(df, left, right):
    headers = {}
    headers["resp_old_headers"] = df.columns[left:right]
    headers["questions"] = rename_headers("Q", headers["resp_old_headers"])
    headers["coefficients"] = [f"C{el[1:]}" for el in headers["questions"]]
    headers["responses"] = [f"R{el[1:]}" for el in headers["questions"]]
    headers["values"] = [f"V{el[1:]}" for el in headers["questions"]]

    return headers


def mapping(from_, to_):
    return {old: new for old, new in zip(from_, to_)}


def append_to_basename(fp: Path, appendix):
    return fp.with_name(f"{fp.stem}{appendix}").with_suffix(fp.suffix)


def text_hist(a, bins=50, width=140):
    h, b = np.histogram(a, bins)

    for i in range(0, bins):
        print(
            "{:12.5f}  | {:{width}s} {}".format(
                b[i], "#" * int(width * h[i] / np.amax(h)), h[i], width=width
            )
        )
    print("{:12.5f}  |".format(b[bins]))


def import_range(response_range):
    try:
        left, right, *_ = response_range
    except ValueError:
        left = response_range[0]
        right = None

    return left, right


def make_out_headers(headers, adjust, anonym):

    headers["grades"] = list(
        itertools.chain.from_iterable(zip(headers["responses"], headers["values"]))
    )

    if adjust:
        headers["totals"] = ["Total", "Total_neg", "Baseline", "Total_adjusted"]
    else:
        headers["totals"] = ["Total", "Total_neg"]

    if anonym:
        headers["ids"] = ["Student ID"]
    else:
        headers["ids"] = ["First name", "Last name", "Student ID"]

    headers["all"] = headers["ids"] + headers["grades"] + headers["totals"]

    return headers


def assess_responses(responses_df, ref_df, headers):

    matches_df = responses_df[headers["questions"]].eq(ref_df.Response)
    matches_df.rename(
        columns=mapping(headers["questions"], to_=headers["responses"]), inplace=True
    )
    return matches_df


def grade(responses_df, matches_df, weights_df, headers):

    grades_df = matches_df.rename(
        columns=mapping(headers["responses"], to_=headers["values"])
    )
    for is_right in weights_df:
        values_df = responses_df[headers["coefficients"]].rename(
            columns=mapping(headers["coefficients"], to_=headers["values"])
        )
        values_df = values_df.applymap(lambda x: weights_df.loc[x, is_right])
        grades_df = grades_df.replace({is_right: np.nan}).combine_first(values_df)

    return grades_df


def compute_results(responses_df, grades_df, matches_df, headers):
    results_df = responses_df.join(matches_df * 1).join(grades_df.astype("int"))
    results_df["Total"] = grades_df.mean(axis="columns").clip(0, 20).round(1)
    results_df["Baseline"] = (matches_df * 18).mean(axis="columns").round(1)
    results_df["Total_neg"] = (
        results_df[headers["values"]]
        .clip(lower=None, upper=0)
        .sum(axis="columns")
        .astype("int")
    )
    results_df["Total_adjusted"] = results_df["Total"].clip(results_df["Baseline"])

    return results_df


def argp():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Input csv file to process."
    )
    parser.add_argument("--to", choices=["csv"], help="Export results to csv file.")
    parser.add_argument(
        "-r",
        "--response-range",
        nargs="+",
        type=int,
        default=[4],
        help="Response range.",
    )
    parser.add_argument(
        "--adjust", action="store_true", help="Export adjusted to baseline total."
    )
    parser.add_argument("--anonym", action="store_true", help="Remove student names.")
    parser.add_argument("--hist", action="store_true", help="Plot histogram.")

    return parser.parse_args()


def main(args):

    fp = Path(args.input)

    left, right = import_range(args.response_range)

    responses_raw_df = pd.read_csv(fp)

    weights_df = (
        pd.read_csv("weights.csv")
        .set_index("Coef")
        .rename(columns={"Right": True, "Wrong": False})
    )

    reference_df = pd.read_csv(append_to_basename(fp, " - Ref")).set_index("Q_id")

    headers = make_headers(responses_raw_df, left, right)

    quest_coeff = list(
        itertools.chain.from_iterable(
            zip(headers["questions"], headers["coefficients"])
        )
    )

    responses_df = responses_raw_df.rename(
        columns=mapping(headers["resp_old_headers"], to_=quest_coeff)
    )

    # Sanity check to verify if response columns were well selected
    assert (responses_df[headers["coefficients"]].dtypes == np.dtype("int")).all()

    matches_df = assess_responses(responses_df, reference_df, headers)

    grades_df = grade(responses_df, matches_df, weights_df, headers)

    results_df = compute_results(responses_df, grades_df, matches_df, headers)

    output_headers = make_out_headers(headers, args.adjust, args.anonym)

    if args.to == "csv":
        results_df[output_headers["all"]].to_csv(
            append_to_basename(fp, " - Results"), float_format="%.1f"
        )

    # To avoid printing ID as float (workaround for .to_markdown() issues)
    results_df["Student ID"] = results_df["Student ID"].astype(str)

    print()
    print(results_df[output_headers["all"]].to_markdown(index=False))
    print()
    print(results_df[output_headers["grades"]].describe().round(2).to_markdown())
    print()
    print(results_df[output_headers["totals"]].describe().round(2).to_markdown())
    print()

    if args.hist:
        results_df.Total_adjusted.hist()
        plt.show()


if __name__ == "__main__":
    try:
        main(argp())
    except KeyboardInterrupt:
        print("\nInterrupted\n")
