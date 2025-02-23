from green_score import GREEN
from utils import flatten_df_test_set, fuse_fine_grained_errors
import pandas as pd
import os
import sys

import argparse

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
model_name = args.model_name

# StanfordAIMI/GREEN-Radlama2-7b
# 1.6204692220687866
# PrunaAI/StanfordAIMI-GREEN-RadLlama2-7b-AWQ-4bit-smashed
# 1.8372502064704894
# PrunaAI/StanfordAIMI-GREEN-RadLlama2-7b-bnb-8bit-smashed
#
# PrunaAI/StanfordAIMI-GREEN-RadLlama2-7b-bnb-4bit-smashed
# 3.74193622469902
# /dataNAS/people/sostm/checkpoints/green_models/radphi2_green_v3/checkpoint-3040


# this is the rexval dataset
data = flatten_df_test_set(pd.read_csv("inference/50_samples_gt_and_candidates.csv"))
mapping = pd.read_csv("inference/mapping.csv")

hyps = data["generated reports"]
refs = data["ground truth reports"]

# initialize the scorer
green_scorer = GREEN(model_name, output_dir=".")
# compute the green analysis
_, _, _, _, result_df = green_scorer(refs, hyps)

# check that the order of the results is correct
assert (
    result_df["reference"] == data["ground truth reports"]
).all(), "Mismatch found between reference and ground truth reports"
assert (
    result_df["predictions"] == data["generated reports"]
).all(), "Mismatch found between predictions and generated reports"

# format for further analyses
df = mapping
mapping["generated response to prompt"] = result_df["green_analysis"]

# save the results
path = f"results_{model_name.rsplit('/')[-1]}/rexval_data"
os.makedirs(path, exist_ok=True)
df.to_csv(f"{path}/responses_test.csv", index=False)

# Use the model_name from the command line
model = model_name.rsplit("/")[-1]

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
