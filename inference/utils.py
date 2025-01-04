import pandas as pd
from tqdm import tqdm


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


if __name__ == "__main__":
    data = pd.read_csv("inference/50_samples_gt_and_candidates.csv")
    flatten_df_test_set(data)
