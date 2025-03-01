# GREEN: Generative Radiology Report Evaluation and Error Notation
[**Project Page**](https://stanford-aimi.github.io/green.html) | 
[**Paper**](https://aclanthology.org/2024.findings-emnlp.21/) | 
[**🤗 Dataset**](https://huggingface.co/datasets/StanfordAIMI/GREEN) | 
[**🤗 Model**](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b)

## Abstract

Evaluating radiology reports is a challenging problem as factual correctness is extremely important due to its medical nature. Existing automatic evaluation metrics either suffer from failing to consider factual correctness (e.g., BLEU and ROUGE) or are limited in their interpretability (e.g., F1CheXpert and F1RadGraph). In this paper, we introduce GREEN (Generative Radiology Report Evaluation and Error Notation), a radiology report generation metric that leverages the natural language understanding of language models to identify and explain clinically significant errors in candidate reports, both quantitatively and qualitatively. Compared to current metrics, GREEN offers a score aligned with expert preferences, human interpretable explanations of clinically significant errors, enabling feedback loops with end-users, and a lightweight open-source method that reaches the performance of commercial counterparts. We validate our GREEN metric by comparing it to GPT-4, as well as to the error counts of 6 experts and the preferences of 2 experts. Our method demonstrates not only a higher correlation with expert error counts but simultaneously higher alignment with expert preferences when compared to previous approaches.

## Installation
Python 3.11 (for now)
```bash
pip install green_score
```
or
```bash
git clone https://github.com/Stanford-AIMI/GREEN.git
cd GREEN
pip install -e .
```
or 
```bash
git clone https://github.com/Stanford-AIMI/GREEN.git
cd GREEN
conda create -n green_score python=3.11
conda activate green_score
pip install -e .
```

## Usage

```python
from green_score import GREEN

refs = [
    "Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
]

model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".")
mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
print(green_score_list)
print(summary)
for index, row in result_df.iterrows():
    print(f"Row {index}:\n")
    for col_name in result_df.columns:
        print(f"{col_name}: {row[col_name]}\n")
    print('-' * 80)
```
For running inference on the ReXVal dataset download `50_samples_gt_and_candidates.csv` to the folder `inference`from [physionet](https://physionet.org/content/rexval-dataset/1.0.0/) and use the following command:
```
torchrun --nproc_per_node=2 inference/run_get_green_analysis.py
```
Adjust the `--nproc_per_node` to your number of GPUS.

The individual error counts and number of matched findings are summarized in `in result_df`. The function that extracts the error counts from a green model output is [parse_error_counts](https://github.com/Stanford-AIMI/GREEN/blob/474ba6ec83238a9ab3343f159032494c3a19b29b/green_score/green.py#L256). The pattern `r'\(\w\)\s.*?:\s(\d+)'` as proposed by claude or gpt, does not yield the same level of correctness in our experiements.

## Benchmark

All scores are reported on the Internal Test (GPT-4 Annotation) dataset. 
  
| Language Model | Data |  |  | $\mathrm{MAE} \pm \mathrm{STD}$ |  | Accuracy $\uparrow$ |  |  |  |  |  |BertScore | Time/sample (secs) | Batch Size* |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | PreRad | CXR | $\mathrm{MM}$ | $\Delta$ Sig. Error $\downarrow$ <br> $7.03 \pm 1.16$ | $\Delta$ Insig. Error $\downarrow$ <br> $0.47 \pm 0.55$ | (a) | (b) | (c) | (d) | (e) | (f) | | |  |
| [Mistral-v0.1 (7B)](https://huggingface.co/StanfordAIMI/GREEN-Mistral-7b) |  | $\checkmark$ |  | $2.60 \pm 1.91$ | $0.87 \pm 0.94$ | 0.13 | 0.31 | 0.62 | 0.59 | 0.48 | 0.67 | 0.80 ± 0.11|3.57 |2,048  |
| [LLaMA-2 (7B](https://huggingface.co/StanfordAIMI/GREEN-Llama2-7b) |  | $\checkmark$ |  | $2.62 \pm 1.25$ | $0.47 \pm 0.52$ | 0.10 | 0.23 | 0.65 | 0.59 | 0.68 | 0.70 | 0.78 ± 0.12| 2.3|2.67|2,048  |
| [LLaMA-2 (7B](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b) | $\checkmark$ | $\checkmark$ |  | $1.54+1.36$ | $0.51 \pm 0.54$ | 0.34 | 0.38 | 0.60 | 0.54 | 0.65 | 0.68 | 0.83 ± 0.24 |2.67 |16|
| [Phi-2 (2.7B)](https://huggingface.co/StanfordAIMI/GREEN-Phi2) |  | $\checkmark$ |  | $2.10 \pm 1.39$ | $0.65 \pm 0.70$ | 0.34 | 0.08 | 0.65 | 0.57 | 0.66 | 0.53 | 0.80 ± 0.11|1.78 |2,048|
| [Phi-2 (2.7B)](https://huggingface.co/StanfordAIMI/GREEN-RadPhi2) | $\checkmark$ | $\checkmark$ |  | $2.08 \pm 1.15$ | $0.55 \pm 0.61$ | 0.19 | 0.18 | 0.62 | 0.57 | 0.62 | 0.61 |  0.84 ± 0.10 | 1.78 | 2,048|
| [Phi-3 (3.8B)](https://huggingface.co/StanfordAIMI/GREEN-Phi-3) |  | $\checkmark$ |  | $2.03 \pm 1.59$ | $0.54 \pm 0.56$ | 0.31 | 0.3 | 0.62 | 0.59 | 0.58 | 0.66 | | | |
| Phi-3 (3.8B)|  | $\checkmark$ | $\checkmark$ | $2.28 \pm 1.22$ | $0.66 \pm 0.71$ | 0.23 | 0.19 | 0.6 | 0.57 | 0.64 | 0.26 | ||  |
| Gemma (2B) |  | $\checkmark$ |  | $2.29 \pm 1.90$ | $0.73 \pm 0.95$ | 0.27 | 0.30 | 0.61 | 0.59 | 0.57 | 0.60 | | | |
| [Gemma (2B)](https://huggingface.co/StanfordAIMI/GREEN-Gemma-2b-MM) |  | $\checkmark$ | $\checkmark$ | $2.25 \pm 1.44$ | $0.55 \pm 0.58$ | 0.32 | 0.28 | 0.57 | 0.55 | 0.66 | 0.19 | | | |
| GREEN GPT-4 |  |  |  | $1.51 \pm 1.29$ | $0.52 \pm 0.55$ | 0.32 | 0.40 | 0.65 | 0.59 | 0.68 | 0.70 | | | |

Please see the [error subcategories for (a)-(f)](https://github.com/Stanford-AIMI/GREEN/blob/4e9e939c06761e13d3d520a6434ee7f5c8cded3e/green_score/green.py#L63).
We are working on benchmarking the new models. 

## Testing
```bash
pytest tests/test_repro.py -s
```

## Reference

```bibtex
@inproceedings{ostmeier-etal-2024-green,
    title = "{GREEN}: Generative Radiology Report Evaluation and Error Notation",
    author = "Ostmeier, Sophie  and
      Xu, Justin  and
      Chen, Zhihong  and
      Varma, Maya  and
      Blankemeier, Louis  and
      Bluethgen, Christian  and
      Md, Arne Edward Michalson  and
      Moseley, Michael  and
      Langlotz, Curtis  and
      Chaudhari, Akshay S  and
      Delbrouck, Jean-Benoit",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.21/",
    doi = "10.18653/v1/2024.findings-emnlp.21",
    pages = "374--390",
}
```
