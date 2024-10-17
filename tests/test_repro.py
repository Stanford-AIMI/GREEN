from green_score import GREEN
import pandas as pd
import difflib
import math

example_count = 5
epsilon = 1e-5

responses_test_df = pd.read_csv("tests/responses_test.csv")
flattened_samples_df = pd.read_csv("tests/RexVal_test.csv")

assert example_count <= len(responses_test_df)

model_name = "StanfordAIMI/GREEN-radllama2-7b"
green_scorer = GREEN(model_name, output_dir=".", cpu=False)

refs = flattened_samples_df["ground truth reports"][:example_count]
hyps = flattened_samples_df["generated reports"][:example_count]
expected_green_score_list = responses_test_df["generated response to prompt"][
    :example_count
]

mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)

count = 0
count_score = 0
for i in range(len(result_df)):
    actual = (
        result_df["green_analysis"][i].strip().replace("</s>", "").replace(":\n", "")
    )
    expected = (
        expected_green_score_list[i].strip().replace("</s>", "").replace(":\n", "")
    )

    # compare strings
    if actual != expected:
        print(f"Test failed on index {i}:")

        diff = difflib.unified_diff(
            expected.splitlines(), actual.splitlines(), lineterm=""
        )
        print("\n".join(diff))  # Print the diff
        count += 1

    # compare scores
    if not math.isclose(
        result_df["green_score"][i],
        green_scorer.compute_green(response=expected_green_score_list[i]),
        rel_tol=epsilon,
    ):
        count_score += 1

print(f"\n\nResponses are the same in {len(result_df) - count}/{ len(result_df)} tests")
print(
    f"\n\nScores are the same in {len(result_df) - count_score}/{ len(result_df)} score tests"
)
