import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from green_score.utils import process_responses, make_prompt, tokenize_batch_as_chat

# A dictionary to store rewards for pairs of reference and hypothesis reports
pair_to_reward_dict = dict()


class GREENModel(nn.Module):
    """
    GREENModel is a neural network model for evaluating radiology reports.

    Args:
        cuda (bool): Whether to use CUDA for GPU acceleration.
        model_id_or_path (str): Path or identifier of the pretrained model.
        do_sample (bool): Whether to sample during generation.
        batch_size (int): Batch size for processing.
        return_0_if_no_green_score (bool): Whether to return 0 if no green score is found.

    Attributes:
        model: Pretrained model for causal language modeling.
        tokenizer: Tokenizer associated with the model.
        categories (list): List of evaluation categories.
        sub_categories (list): List of subcategories for error evaluation.
    """

    def __init__(
            self,
            cuda,
            model_id_or_path,
            do_sample=False,
            batch_size=4,
            return_0_if_no_green_score=True,
    ):
        super().__init__()
        self.cuda = cuda
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.return_0_if_no_green_score = return_0_if_no_green_score
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            trust_remote_code=True,
            device_map={"": "cuda:{}".format(torch.cuda.current_device())} if cuda else "cpu",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n{% endfor %}"

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]

        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

    def get_response(self, input_ids, attention_mask):
        """
        Generates responses using the model and processes them.

        Args:
            input_ids (Tensor): Input IDs for the model.
            attention_mask (Tensor): Attention mask for the input IDs.

        Returns:
            tuple: Processed response list and output IDs.
        """
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.do_sample,
            max_length=2048,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response_list = process_responses(responses)

        return response_list, outputs

    def parse_error_counts(self, text, category, for_reward=False):
        """
        Parses error counts from the generated text for a specific category.

        Args:
            text (str): Text to parse for error counts.
            category (str): Category of errors to parse.

        Returns:
            tuple: Sum of counts and list of subcategory counts.
        """
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if self.return_0_if_no_green_score:
                return sum_counts, sub_counts
            else:
                return None, [None for i in range(6)]

        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts

        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def compute_green(self, response):
        """
        Computes the green score based on significant clinical errors and matched findings.

        Args:
            response (str): Generated response to evaluate.

        Returns:
            float: Computed green score.
        """
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the model, computing green scores for input batch.

        Args:
            input_ids (Tensor): Input IDs for the model.
            attention_mask (Tensor): Attention mask for the input IDs.

        Returns:
            tuple: Tensor of green scores and output IDs.
        """
        if self.cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        reward_model_responses, output_ids = self.get_response(input_ids, attention_mask)

        greens = [self.compute_green(response) for response in reward_model_responses]
        greens = [green for green in greens if green is not None]

        return torch.tensor(greens, dtype=torch.float), output_ids


class GREEN(nn.Module):
    """
    GREEN is a wrapper model for GREENModel, handling batching and aggregation.

    Args:
        cuda (bool): Whether to use CUDA for GPU acceleration.

    Attributes:
        model: GREENModel instance for evaluation.
        tokenizer: Tokenizer associated with the model.
    """

    def __init__(self, cuda, **kwargs):
        super().__init__()
        self.cuda = cuda
        self.model = GREENModel(cuda, **kwargs)
        self.tokenizer = self.model.tokenizer
        if self.cuda:
            print("Using {} GPUs!".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

    def forward(self, refs, hyps):
        """
        Forward pass for the model, computing green scores for pairs of reference and hypothesis reports.

        Args:
            refs (list): List of reference reports.
            hyps (list): List of hypothesis reports.

        Returns:
            tuple: Mean green score, tensor of green scores, and list of processed responses.
        """
        assert len(refs) == len(hyps)

        with torch.no_grad():
            pairs_to_process = []
            indices_to_process = []
            final_scores = torch.zeros(len(refs))

            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if (ref, hyp) in pair_to_reward_dict:
                    final_scores[i] = pair_to_reward_dict[(ref, hyp)]
                else:
                    pairs_to_process.append((ref, hyp))
                    indices_to_process.append(i)

            if pairs_to_process:
                batch = [make_prompt(ref, hyp) for ref, hyp in pairs_to_process]
                batch = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch]
                batch = tokenize_batch_as_chat(self.tokenizer, batch)

                greens_tensor, output_ids = self.model(batch['input_ids'], batch['attention_mask'])

                if len(greens_tensor) == len(pairs_to_process):
                    for i, (ref, hyp) in enumerate(pairs_to_process):
                        score = greens_tensor[i]
                        pair_to_reward_dict[(ref, hyp)] = score
                        final_scores[indices_to_process[i]] = score
                else:
                    print("An inconsistency was detected in processing pairs.")

            mean_green = final_scores.mean()
            responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            return mean_green, final_scores, process_responses(responses)


if __name__ == '__main__':
    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN",
        do_sample=False,  # should be always False
        batch_size=32,
        return_0_if_no_green_score=True,
        cuda=True,
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

    mean_green, greens, text = model(refs=refs, hyps=hyps)
    print("Mean reward for the given examples is: ", mean_green)
    print("Array of rewards for the given examples is: ", greens)
    print(text[0])
