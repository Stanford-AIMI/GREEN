from green_score import GREEN
import pandas as pd
import difflib
import math
import json
import warnings

# please commented out 2 lines below if you are using an environment with different versions
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# run with
# pytest tests/test_repro.py -s

epsilon = 1e-5

model_name = "StanfordAIMI/GREEN-radllama2-7b"
green_scorer = GREEN(model_name, output_dir=".", cpu=False, compute_summary_stats=True)


refs = [
    "1.  Increased opacification of the bilateral bases to the mid lung fields, which most likely represents pulmonary edema, however underlying infection cannot be excluded.    2.  Stable cardiomegaly.   3.  Small-to-moderate bilateral pleural effusions.",
    "1.  Left lower lobe pneumonia with small left parapneumonic effusion. 2.  New focal airspace opacities in the right lung apex and right lung base may represent superimposition of normal structures.  Attention is recommended on followup.  Findings were discussed by Dr. ___ with Dr. ___ ___ medicine via telephone at 10:50am on ___.",
    "In comparison with the study of ___, there are worsened lung volumes. Monitoring and support devices are unchanged.  Opacification at the left base is consistent with substantial volume loss in the lower lobes and pleural effusion. No definite pulmonary vascular congestion.  The left lung is essentially clear.",
    "Stable chest findings, no evidence of new acute infiltrates.",
    "Focal peripheral right upper lobe noduular opacity appears slightly more prominent than on prior studies , possibly due to overlap of the right scapula . However , further evaluation with a chest CT may be helpful to more fully characterize this region and to exclude the possibility of a slowly growing lung adenocarcinoma at this site . Findings and recommendation were discussed by telephone with Dr . ___ at 11 a . m . on ___ at the time of discovery .",
]
hyps = [
    "1. Low lung volumes.  Mild interstitial pulmonary edema, improved from the previous exam.    2.  Near-complete interval resolution of bilateral pleural effusions since ___.    3.  Prominent mediastinal silhouette is most likely due to low lung volumes and patient's positioning.  A repeat conventional PA and lateral radiographs will be helpful, when tolerated.",
    "1.  Large right hilar lung mass and radiation fibrosis.  Additional post-obstructive pneumonia in the right upper and lower lobes is possible but hard to delineate. 2.  New left retrocardiac opacity, small left effusion, and pleural thickening.  Findings were discussed with ___, RN, via telephone at ___ and again with Dr ___ at ___.",
    "In comparison with the study of ___, the monitoring and support devices are unchanged.  Opacification at the right base is unchanged, again consistent with collapse of the middle and lower lobes.  The left lung remains clear.",
    "In comparison the prior study there is a 10cm opacity in the right lower lobe. No evidence of new acute pneumonia.",
    "Irregularly marginated 3-cm mass in the lingula has grown since prior studies. Although previously attributed to round atelectasis, its  growth and margins raise the potential concern for a slowly growing lung adenocarcinoma.  CT of the chest is recommended for further evaluation of this finding.  These findings were discussed with Dr. ___ via phone at 3:05 p.m. by ___.",
]
expected_green_score_list = [
    "\nThe candidate report has several discrepancies when compared to the reference report. The candidate report mentions low lung volumes and mild interstitial pulmonary edema, which are not mentioned in the reference report. The reference report mentions increased opacification of the bilateral bases to the mid lung fields, stable cardiomegaly, and small-to-moderate bilateral pleural effusions, none of which are mentioned in the candidate report. The candidate report also mentions a near-complete interval resolution of bilateral pleural effusions, which contradicts the reference report's mention of small-to-moderate bilateral pleural effusions.\n\n[Clinically Significant Errors]:\n(a) False report of a finding in the candidate: 2. Low lung volumes; Mild interstitial pulmonary edema.\n(b) Missing a finding present in the reference: 3. Increased opacification of the bilateral bases to the mid lung fields; Stable cardiomegaly; Small-to-moderate bilateral pleural effusions.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 1. Near-complete interval resolution of bilateral pleural effusions.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Clinically Insignificant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Matched Findings]:\n0.",
    "\nThe candidate report has several clinically significant errors. It falsely reports a large right hilar lung mass and radiation fibrosis, which are not mentioned in the reference report. It also misidentifies the location of the pneumonia, stating it is in the right upper and lower lobes, while the reference report mentions it in the left lower lobe. The candidate report also misses the finding of new focal airspace opacities in the right lung apex and right lung base. The candidate report also misidentifies the location of the left retrocardiac opacity, which is not mentioned in the reference report. \n\n[Clinically Significant Errors]:\n(a) False report of a finding in the candidate: 2. Large right hilar lung mass; Radiation fibrosis.\n(b) Missing a finding present in the reference: 1. New focal airspace opacities in the right lung apex and right lung base.\n(c) Misidentification of a finding's anatomic location/position: 2. Pneumonia in the right upper and lower lobes; Left retrocardiac opacity.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Clinically Insignificant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Matched Findings]:\n1. Small left effusion.",
    "\nThe candidate report has several discrepancies when compared to the reference report. The candidate report incorrectly identifies the location of the opacification, stating it is at the right base instead of the left. It also fails to mention the worsened lung volumes and the substantial volume loss in the lower lobes and pleural effusion. Furthermore, the candidate report incorrectly states that the opacification is unchanged, which contradicts the reference report's finding of worsened lung volumes.\n\n[Clinically Significant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 2. Worsened lung volumes; Substantial volume loss in the lower lobes and pleural effusion.\n(c) Misidentification of a finding's anatomic location/position: 1. Opacification at the right base instead of the left.\n(d) Misassessment of the severity of a finding: 1. Incorrectly stated that the opacification is unchanged.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Clinically Insignificant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Matched Findings]:\n2. The monitoring and support devices are unchanged; The left lung remains clear.",
    "\nThe candidate report mentions a 10cm opacity in the right lower lobe, which is not mentioned in the reference report. This is a clinically significant error as it falsely reports a finding that is not present in the reference report. The candidate report correctly mentions that there is no evidence of new acute pneumonia, which matches the reference report's finding of no evidence of new acute infiltrates.\n\n[Clinically Significant Errors]:\n(a) False report of a finding in the candidate: 1. 10cm opacity in the right lower lobe.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Clinically Insignificant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Matched Findings]:\n1. No evidence of new acute pneumonia.",
    "\nThe candidate report has several significant discrepancies when compared to the reference report. The candidate report incorrectly identifies the location of the lung abnormality, stating it is in the lingula instead of the right upper lobe. It also incorrectly describes the abnormality as a 3-cm mass, whereas the reference report describes it as a nodular opacity. The candidate report also incorrectly attributes the growth of the abnormality to round atelectasis, whereas the reference report attributes it to the overlap of the right scapula. \n\n[Clinically Significant Errors]:\n(a) False report of a finding in the candidate: 1. The candidate report falsely attributes the growth of the lung abnormality to round atelectasis.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 1. The candidate report incorrectly identifies the location of the lung abnormality as the lingula instead of the right upper lobe.\n(d) Misassessment of the severity of a finding: 1. The candidate report incorrectly describes the lung abnormality as a 3-cm mass, whereas the reference report describes it as a nodular opacity.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Clinically Insignificant Errors]:\n(a) False report of a finding in the candidate: 0.\n(b) Missing a finding present in the reference: 0.\n(c) Misidentification of a finding's anatomic location/position: 0.\n(d) Misassessment of the severity of a finding: 0.\n(e) Mentioning a comparison that isn't in the reference: 0.\n(f) Omitting a comparison detailing a change from a prior study: 0.\n\n[Matched Findings]:\n1. Both reports recommend further evaluation with a chest CT to better characterize the lung abnormality and exclude the possibility of a slowly growing lung adenocarcinoma.",
]
mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)


# # save results
# result_df.to_csv("results_df.csv", index=False)
# # save green_score_list
# with open("output.json", "w") as f:
#     json.dump(list(result_df["green_analysis"]), f)


# Test cases
def test_green_score_responses():

    for i in range(len(result_df)):
        actual = (
            result_df["green_analysis"][i]
            .strip()
            .replace("</s>", "")
            .replace(":\n", "")
        )
        expected = (
            expected_green_score_list[i].strip().replace("</s>", "").replace(":\n", "")
        )

        # Assert that the actual response matches the expected response
        assert (
            actual == expected
        ), f"Mismatch at index {i}:\n{get_diff(expected, actual)}"


def test_green_score_values():

    for i in range(len(result_df)):
        actual_score = result_df["green_score"][i]
        expected_score = green_scorer.compute_green(
            response=expected_green_score_list[i]
        )
        assert math.isclose(
            actual_score, expected_score, rel_tol=epsilon
        ), f"Mismatch in scores at index {i}: Actual {actual_score}, Expected {expected_score}"


def get_diff(expected, actual):
    diff = difflib.unified_diff(expected.splitlines(), actual.splitlines(), lineterm="")
    return "\n".join(diff)
