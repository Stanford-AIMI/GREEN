from src.green import compute
import time
import torch.distributed as dist

refs = [
    "Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis.", 
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are changed. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis.",     
    "Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are changed. ",
]
hyps = [
    "Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are changed. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis.",    
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs.There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. Advanced central venous congestion and interstitial pulmonary edema. Big bilateral pleural effusions are changed. There is discontinued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly increased, with residual left lower lobe atelectasis.", 
]

model_name = "StanfordAIMI/GREEN-RadLlama2-7b"
# "StanfordAIMI/GREEN-RadLlama2-7b" Seconds per example:  2.67
# "StanfordAIMI/GREEN-Phi2" Seconds per example:  1.78
# "StanfordAIMI/GREEN-Mistral-7b" Seconds per example: 3.57

compute(model_name, refs, hyps, output_dir=".")