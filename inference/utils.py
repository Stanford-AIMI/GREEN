import pandas as pd
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from tqdm import tqdm
import os
from green_score import GREEN
import argparse


def flatten_df_test_set(data):
    # study_number or study_id
    if "study_id" in data.columns:
        id = "study_id"
    else:
        id = "study_number"

    data[id] = list(range(len(data)))
    # otherwise it throws an error if not str
    data[id] = data[id].astype(str)
    types = ["radgraph", "bertscore", "s_emb", "bleu"]

    origin = []
    study_number = []
    candidate_report = []
    grount_truth_report = []

    for row in tqdm(data.iterrows()):
        origin.extend(types)
        study_number.extend([row[1][id]] * len(types))
        candidate_report.extend([row[1][i] for i in types])
        grount_truth_report.extend([row[1]["gt_report"]] * len(types))

    assert (
        len(origin) == len(study_number) == len(candidate_report)
    ), f"Lengths do not match: origin {len(origin)}, study_number {len(study_number)}, candidate_report {len(candidate_report)}"

    df = pd.DataFrame(
        {
            "study_number": study_number,
            "origin": origin,
            "generated reports": candidate_report,
            "ground truth reports": grount_truth_report,
        }
    )
    return df


def plot_correlation_to_radiologist(list1, list2, filename: str):
    # Compute correlation
    corr_pearson = np.corrcoef(list1, list2)[0, 1]
    print(f"Correlation for {filename}: {corr_pearson:.2f}")
    corr_kendalltau, p_value = kendalltau(list1, list2)
    print(f"Kendalltau for {filename}: {corr_kendalltau:.2f}")
    print("P-value:", p_value)

    plt.scatter(list1, list2)
    plt.xlabel("Radiologist Rating")
    plt.ylabel("LLM Rating")
    plt.title(f"Correlation for {filename.split('.')[0]}")
    plt.plot([0, max(list2)], [0, max(list2)], "k-")

    # Add correlation to the plot
    plt.text(4, 0.5, f"Correlation: {corr_pearson:.2f}")
    plt.text(4, 0, f"Kendalltau: {corr_kendalltau:.2f}")

    plt.savefig(filename)
    plt.close()


def get_worst_cases(radiolgist_rating, llm_rating, llm_worst):

    llm_rating["error_diff"] = abs(
        radiolgist_rating["num_sig_errors"] - llm_rating["num_sig_errors"]
    )
    llm_rating["radiologist_num_sig_errors"] = radiolgist_rating["num_sig_errors"]
    llm_rating["radiologist_num_insig_errors"] = radiolgist_rating["num_insig_errors"]
    llm_rating["radiologist_total_num_errors"] = radiolgist_rating["total_num_errors"]

    # Sort by error_diff in descending order and select the top 10
    worst_cases = llm_rating.sort_values("error_diff", ascending=False).head(10)

    # Save to a new CSV file
    worst_cases.to_csv(llm_worst, index=False)


def sort_df_by_composite_key(
    df_to_be_sorted, df_to_sort_by, study_number="study_number", origin="origin"
):
    # Create a composite key in both dataframes
    df_to_sort_by["composite_key"] = (
        df_to_sort_by[study_number].astype(str) + "_" + df_to_sort_by[origin]
    )
    df_to_be_sorted["composite_key"] = (
        df_to_be_sorted[study_number].astype(str) + "_" + df_to_be_sorted[origin]
    )

    # Create a dictionary that maps each composite_key to its position in df_to_sort_by
    order_dict = {k: v for v, k in enumerate(df_to_sort_by["composite_key"])}

    # Create a new column in df_to_be_sorted that contains the position of each composite_key
    df_to_be_sorted["order"] = df_to_be_sorted["composite_key"].map(order_dict)

    # Sort df_to_be_sorted by the new column
    df_to_be_sorted = df_to_be_sorted.sort_values("order")

    # Drop the order and composite_key columns
    df_to_be_sorted = df_to_be_sorted.drop(["order", "composite_key"], axis=1)

    return df_to_be_sorted


def compute_correlation(df):
    per_case_df = df.groupby(["study_number", "origin"]).sum().reset_index()
    sig_per_case = per_case_df["sig_llm_errors"]
    insig_per_case = per_case_df["insig_llm_errors"]
    matched_findings = per_case_df["matched_findings"] // 6
    import ipdb

    ipdb.set_trace()
    green_per_case = matched_findings / (sig_per_case + matched_findings)
    summed_error_rater = per_case_df["total_mean_errors"]
    sig_correlation = plot_correlation_to_radiologist(sig_per_case, summed_error_rater)
    total_correlation = plot_correlation_to_radiologist(
        sig_per_case + insig_per_case, summed_error_rater
    )
    green_correlation = plot_correlation_to_radiologist(
        green_per_case, summed_error_rater
    )

    # save the results for correlation analysis
    corr_df = pd.DataFrame(
        {
            "corr": [sig_correlation[0], total_correlation[0], green_correlation[0]],
            "p_value": [sig_correlation[1], total_correlation[1], green_correlation[1]],
        },
    )
    corr_df.to_csv("corr_df.csv", index=False)


def fuse_fine_grained_errors(llm_rating_df, radio_df, path):
    # Sort the dataframes by the composite key
    llm_rating_df = sort_df_by_composite_key(llm_rating_df, radio_df)
    sig = []
    insig = []
    error_category = []
    matched_findings = []
    inf = GREEN(cpu=True)
    for i in tqdm(llm_rating_df["generated response to prompt"]):
        # significant errors
        _, sig_array = inf.parse_error_counts(text=i, category=inf.categories[0])
        sig.extend(sig_array)
        # insignificant errors
        _, insig_array = inf.parse_error_counts(text=i, category=inf.categories[1])
        insig.extend(insig_array)
        # matched findings
        matched_finding, _ = inf.parse_error_counts(text=i, category=inf.categories[2])
        matched_findings.extend([matched_finding] + [0] * (len(sig_array) - 1))
        # error category
        error_category.extend(
            list(range(1, len(sig_array) + 1))
        )  # because ReXVal starts at 1

    radio_df["sig_llm_errors"] = sig
    radio_df["insig_llm_errors"] = insig
    radio_df["matched_findings"] = matched_findings

    assert (
        len(radio_df)
        == len(sig)
        == len(insig)
        == len(error_category)
        == len(matched_findings)
    ), f"Lengths do not match. Here are the lengths of the arrays: \n radio_df: {len(radio_df)} \n sig: {len(sig)} \n insig: {len(insig)} \n error_category: {len(error_category)} \n matched_findings: {len(matched_findings)}"

    import ipdb

    ipdb.set_trace()
    compute_correlation(radio_df)

    # save the results for correlation analysis
    # generated response to prompt plus scores
    save_df = pd.DataFrame(
        columns=[
            "study_number",
            "origin",
            "error_category",
            "sig_llm_errors",
            "insig_llm_errors",
            "matched_findings",
        ]
    )
    save_df["study_number"] = radio_df["study_number"]
    save_df["origin"] = radio_df["origin"]
    save_df["error_category"] = error_category
    save_df["sig_llm_errors"] = sig
    save_df["insig_llm_errors"] = insig
    save_df["matched_findings"] = matched_findings

    new_path = path.rsplit("/", 1)[0] + f"/results_response_scores.csv"
    print("Saving results to csv")
    print(new_path)
    save_df.to_csv(new_path)

    assert (
        len(radio_df) == len(sig) == len(insig) == len(error_category)
    ), "Lengths do not match"
    assert radio_df["origin"].equals(save_df["origin"]), "Origin do not match"
    assert radio_df["study_number"].equals(
        save_df["study_number"]
    ), "Study numbers do not match"

    radio_df["sig_llm_errors_difference"] = abs(sig - radio_df["sig_mean_errors"])
    radio_df["insig_llm_errors_difference"] = abs(insig - radio_df["insig_mean_errors"])

    radio_df["total_llm_errors"] = [s + i for s, i in zip(sig, insig)]
    radio_df["total_llm_errors_difference"] = (
        radio_df["sig_llm_errors_difference"] + radio_df["insig_llm_errors_difference"]
    )

    radio_df = find_closest_radiologist(radio_df, sig="sig")
    radio_df = find_closest_radiologist(radio_df, sig="insig")

    print("Significant Error")
    compute_metrics(radio_df, path, sig="sig")
    # make dataframe from dict
    print("Insignificant Error")
    compute_metrics(radio_df, path, sig="insig")
    print("Total Error")
    compute_metrics(radio_df, path, sig="total")
    new_path = path.rsplit("/", 1)[0]
    radio_df.drop(
        columns=[
            "composite_key",
        ],
        inplace=True,
    )
    radio_df.to_csv(f"{new_path}/total_mean_{path.split('/')[-1]}_ratings.csv")


def find_closest_radiologist(df, sig="sig"):

    columns_to_compare = ["0", "1", "2", "3", "4", "5"]

    if sig == "insig":
        for i in range(len(columns_to_compare)):
            columns_to_compare[i] = columns_to_compare[i] + ".1"

    # Initialize new columns
    df[f"{sig}_closest_column"] = np.nan
    df[f"{sig}_closest_value"] = np.nan
    # Convert columns to numeric, assuming 'columns_to_compare' is a list of column names
    for col in columns_to_compare:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )  # 'coerce' converts non-convertible values to NaN
    df[f"{sig}_llm_errors"] = pd.to_numeric(df[f"{sig}_llm_errors"], errors="coerce")

    # Then perform the operation
    for idx, row in df.iterrows():
        difference_df = (row[columns_to_compare] - row[f"{sig}_llm_errors"]).abs()
        closest_column = pd.Series(list(difference_df)).idxmin()
        # get the correct column name
        colum_name = columns_to_compare[closest_column]
        closest_value = row[str(colum_name)]
        df.loc[idx, f"{sig}_closest_column"] = closest_column
        df.loc[idx, f"{sig}_closest_value"] = closest_value

    df[f"{sig}_nearest_llm_error_difference"] = abs(
        df[f"{sig}_closest_value"] - df[f"{sig}_llm_errors"]
    )
    assert (
        f"{sig}_nearest_llm_error_difference" in df.columns
    ), f"{sig}_nearest_llm_error_difference not in columns"

    return df


def compute_metrics(df, path, sig="sig"):
    print(df.head())
    print(df.columns)
    if sig == "total":
        df["total_nearest_llm_error_difference"] = (
            df["sig_nearest_llm_error_difference"]
            + df["insig_nearest_llm_error_difference"]
        )

    grouped = df.groupby("error_category")
    grouped_nearest = df.groupby("error_category")
    summed_errors_per_category = grouped[f"{sig}_llm_errors_difference"].sum()
    summed_errors_per_category_nearest = grouped_nearest[
        f"{sig}_nearest_llm_error_difference"
    ].sum()

    # save sum mean error and std in a dataframe
    df_sum = pd.DataFrame(columns=["Mean", "Std", "Mean nearest", "Std nearest"])
    df_sum["Mean"] = [
        df.groupby(["study_number", "origin"])[f"{sig}_llm_errors_difference"]
        .sum()
        .mean()
    ]
    df_sum["Std"] = [
        df.groupby(["study_number", "origin"])[f"{sig}_llm_errors_difference"]
        .sum()
        .std()
    ]
    df_sum["Mean nearest"] = [
        df.groupby(["study_number", "origin"])[f"{sig}_nearest_llm_error_difference"]
        .sum()
        .mean()
    ]
    df_sum["Std nearest"] = [
        df.groupby(["study_number", "origin"])[f"{sig}_nearest_llm_error_difference"]
        .sum()
        .std()
    ]
    df_sum.to_csv(f"{path.rsplit('/', 1)[0]}/{sig}_error_sum.csv")

    # saving results as csv
    inferer = GREEN(cpu=True)

    df_metrics_nearest = pd.DataFrame(
        columns=["Category", "Sum", "MAE", "MAE Std", "Accuracy"]
    )
    df_metrics = pd.DataFrame(columns=["Category", "Sum", "MAE", "MAE Std", "Accuracy"])

    df_metrics["Category"] = inferer.sub_categories
    df_metrics["Sum"] = summed_errors_per_category.values
    df_metrics["MAE"] = grouped[f"{sig}_llm_errors_difference"].mean().values
    df_metrics["MAE Std"] = grouped[f"{sig}_llm_errors_difference"].std().values
    df_metrics["Accuracy"] = (
        1
        - grouped[f"{sig}_llm_errors_difference"]
        .apply(lambda x: x.ne(0).sum())
        .reset_index(drop=True)
        / grouped[f"{sig}_llm_errors_difference"].count().values
    )

    df_metrics_nearest["Category"] = inferer.sub_categories
    df_metrics_nearest["Sum"] = summed_errors_per_category_nearest.values
    df_metrics_nearest["MAE"] = (
        grouped_nearest[f"{sig}_nearest_llm_error_difference"].mean().values
    )
    df_metrics_nearest["MAE Std"] = (
        grouped_nearest[f"{sig}_nearest_llm_error_difference"].std().values
    )
    df_metrics_nearest["Accuracy"] = (
        1
        - grouped_nearest[f"{sig}_nearest_llm_error_difference"]
        .apply(lambda x: x.ne(0).sum())
        .reset_index(drop=True)
        / grouped_nearest[f"{sig}_nearest_llm_error_difference"].count().values
    )

    df_metrics = df_metrics.round(decimals=2)
    df_metrics_nearest = df_metrics_nearest.round(decimals=2)

    new_path = path.rsplit("/", 1)[0]
    df_metrics.to_csv(f"{new_path}/{sig}_error_metrics.csv")
    df_metrics_nearest.to_csv(f"{new_path}/{sig}_nearest_error_metrics.csv")


if __name__ == "__main__":
    # data = pd.read_csv("inference/50_samples_gt_and_candidates.csv")
    # flatten_df_test_set(data)
    # Setup argparse for command-line arguments
    parser = argparse.ArgumentParser(description="Run the GREEN scoring model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Specify the model name (e.g., 'PrunaAI/StanfordAIMI-GREEN-RadLlama2-7b-bnb-4bit-smashed').",
    )

    # Parse arguments
    args = parser.parse_args()

    # Use the model_name from the command line
    model = args.model_name

    root = "."

    print(f"Analyzing model {model}")
    root_model = f"{root}/results_{model}"
    response_file_path = f"{root_model}/rexval_data/responses_test.csv"
    radio_path = f"inference/total_mean_error_radiologist_categories.csv"

    if not os.path.exists(response_file_path):
        ValueError(f"File {response_file_path} does not exist")

    ######### for quick test set analysis ############
    # create mapping
    original_df = pd.read_csv(f"inference/50_samples_gt_and_candidates.csv")
    study_id = original_df["study_id"]
    study_number = [i for i in range(len(study_id))]
    # make dict
    study_dict = dict(zip(study_id, study_number))

    print("looking at file", response_file_path)
    data = pd.read_csv(response_file_path)
    if not "study_number" in data.columns:
        data["study_number"] = data["study_id"].map(study_dict)
    rater = pd.read_csv(radio_path)

    fuse_fine_grained_errors(data, rater, response_file_path)
