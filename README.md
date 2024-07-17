# GREEN: Generative Radiology Report Evaluation and Error Notation
[[Project Page](https://stanford-aimi.github.io/green.html)][[Paper](https://arxiv.org/pdf/2405.03595)]

## Abstract

Evaluating radiology reports is a challenging problem as factual correctness is extremely important due to its medical nature. Existing automatic evaluation metrics either suffer from failing to consider factual correctness (e.g., BLEU and ROUGE) or are limited in their interpretability (e.g., F1CheXpert and F1RadGraph). In this paper, we introduce GREEN (Generative Radiology Report Evaluation and Error Notation), a radiology report generation metric that leverages the natural language understanding of language models to identify and explain clinically significant errors in candidate reports, both quantitatively and qualitatively. Compared to current metrics, GREEN offers a score aligned with expert preferences, human interpretable explanations of clinically significant errors, enabling feedback loops with end-users, and a lightweight open-source method that reaches the performance of commercial counterparts. We validate our GREEN metric by comparing it to GPT-4, as well as to the error counts of 6 experts and the preferences of 2 experts. Our method demonstrates not only a higher correlation with expert error counts but simultaneously higher alignment with expert preferences when compared to previous approaches.

## Installation

```bash
pip install green-score
```

## Usage

```python
from green_score import GREEN

model = GREEN(
    model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
    do_sample=False,  # should be always False
    batch_size=16,
    return_0_if_no_green_score=True,
    cuda=True,
    return_summary=True # set to true for green summary
)

refs = [
    "Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
]

mean_green, greens, explanations = model(refs=refs, hyps=hyps)
print("Mean reward for the given examples is: ", mean_green)
print("Array of rewards for the given examples is: ", greens)
print(explanations[0]) # LLM output for first pair

```

## Benchmark

All scores are reported on the Internal Test (GPT-4 Annotation) dataset. 
| Model                                                                | MAE of Significant Error Counts (↓) | MAE of Insignificant Error Counts (↓) | MAE of Matched Findings (↓) | BERTScore (↑) | Time/sample (secs) | Batch Size* |
|-------------------------------------------------------------------------------------------|------|------|------|------|-----|----|
| [StanfordAIMI/GREEN-RadLlama2-7b](https://huggingface.co/StanfordAIMI/GREEN-RadLlama2-7b) | 0.70 | 0.20 | 0.29 | 0.83 | 2.3 | 16 |
| [StanfordAIMI/GREEN-RadPhi2](https://huggingface.co/StanfordAIMI/GREEN-RadPhi2) | 0.63 | 0.18 | 0.26 | 0.84 |  |  |
| [StanfordAIMI/GREEN-Phi2](https://huggingface.co/StanfordAIMI/GREEN-Phi2) | 0.84 | 0.20 | 0.34 | 0.80 |  |  |
| [StanfordAIMI/GREEN-Llama2-7b](https://huggingface.co/StanfordAIMI/GREEN-Llama2-7b) | 1.35 | 0.15 | 1.62 | 0.78 |  |  |
| [StanfordAIMI/GREEN-Mistral-7b](https://huggingface.co/StanfordAIMI/GREEN-Mistral-7b) | 0.97 | 0.22 | 0.44 | 0.80 |  |  |
* A100 (40GB)

## Reference

```bibtex
@article{ostmeier2024green,
    title={GREEN: Generative Radiology Report Evaluation and Error Notation},
    author={Ostmeier, Sophie and Xu, Justin and Chen, Zhihong and Varma, Maya and Blankemeier, Louis and Bluethgen, Christian and Michalson, Arne Edward and Moseley, Michael and Langlotz, Curtis and Chaudhari, Akshay S and others},
    journal={arXiv preprint arXiv:2405.03595},
    year={2024}
}
```
