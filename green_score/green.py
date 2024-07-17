import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from green_score.utils import process_responses, make_prompt, tokenize_batch_as_chat, truncate_to_max_len, flatten_values_lists_of_list_dicts_to_dict, compute_largest_cluster
import numpy as np

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
        self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

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

    def parse_error_counts(self, text, category):
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
    
    def parse_error_sentences(self, response, category):
        """
        Parses error sentences from a given response based of the specified category. Extracts sentences associated with each sub-categories and returns them in a dict format.

        Args:
            text (str): The input text containing error information.
            category (str): The category to parse within the text.

        Returns:
            dict: A dictionary where keys are sub-categories and values are lists of sentences associated with those sub-categories. If the category is "Matched Findings", returns a list of sentences directly.
        """
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        sub_category_dict_sentences = {}
        for position, sub_category in enumerate(self.sub_categories):
            # need to loop over all matches, because the sub_categories are not always in the same order
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    # If the sub_category is found, add to dictionary
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )
        
        return sub_category_dict_sentences
    
    def compute_sentences(self,response):
        # for now we only look at the significant clinical errors, which is the first category
        return self.parse_error_sentences(response, self.categories[0])
    
    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            print(f"Computing clusters for {i}")
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict
    
    def compute_accuracy(self, responses):
        """
        Computes the accuracy for each subcategory based on significant clinical errors and matched findings.

        Args:
            responses (list): Generated responses to evaluate.

        Returns:
            dict: accurarcies per subcategory.
        """
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            # compute the accuracy for each subcategory
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc
    
    def compute_summary(self, mean_green, std_green, responses):
        """
        Makes green summary.

        Args:
            mean_green (int): grean average.
            mean_std (int): grean std.
            responses (list): list of green model responses (str)

        Returns:
            str: green summary.
        """
        representative_sentences = self.get_representative_sentences(responses)
        accuracies = self.compute_accuracy(responses)

        summary = f"[Summary]: Green average {mean_green} and standard variation {std_green} \\n [Clinically Significant Errors Analyses]:\n (a) False report of a finding in the candidate: {accuracies[self.sub_categories[0]]}. {representative_sentences[self.sub_categories[0]]} \n (b) Missing a finding present in the reference: {accuracies[self.sub_categories[1]]}. {representative_sentences[self.sub_categories[1]]} \n (c) Misidentification of a finding's anatomic location/position: {accuracies[self.sub_categories[2]]}. {representative_sentences[self.sub_categories[2]]} \n (d) Misassessment of the severity of a finding: {accuracies[self.sub_categories[3]]}. {representative_sentences[self.sub_categories[3]]} \n (e) Mentioning a comparison that isn't in the reference: {accuracies[self.sub_categories[4]]}. {representative_sentences[self.sub_categories[4]]} \n (f) Omitting a comparison detailing a change from a prior study: {accuracies[self.sub_categories[5]]}. {representative_sentences[self.sub_categories[5]]}."

        return summary

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

    def __init__(self, cuda, max_len=200, return_summary=False, **kwargs):
        super().__init__()
        self.cuda = cuda
        self.max_len = max_len
        self.model = GREENModel(cuda, **kwargs)
        self.tokenizer = self.model.tokenizer
        self.return_summary = return_summary
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

        refs = truncate_to_max_len(refs, self.max_len)
        hyps = truncate_to_max_len(hyps, self.max_len)

        with torch.no_grad():
            pairs_to_process = []
            final_scores = torch.zeros(len(refs))
            output_ids_dict = {}

            # Iterate over ref-hyp pairs and populate final_scores and pairs_to_process
            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if (ref, hyp) in pair_to_reward_dict:
                    final_scores[i], output_ids = pair_to_reward_dict[(ref, hyp)]
                    output_ids_dict[i] = output_ids
                else:
                    pairs_to_process.append((ref, hyp, i))

            if pairs_to_process:
                batch = [make_prompt(ref, hyp) for ref, hyp, _ in pairs_to_process]
                batch = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch]
                batch = tokenize_batch_as_chat(self.tokenizer, batch)

                greens_tensor, output_ids = self.model(batch['input_ids'], batch['attention_mask'])

                if len(greens_tensor) == len(pairs_to_process):
                    for (ref, hyp, idx), score, out_id in zip(pairs_to_process, greens_tensor, output_ids):
                        pair_to_reward_dict[(ref, hyp)] = (score, out_id)
                        final_scores[idx] = score
                        output_ids_dict[idx] = out_id
                else:
                    print("An inconsistency was detected in processing pairs.")

            responses = [output_ids_dict[i] for i in range(len(refs))]
            responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            responses = process_responses(responses)

            mean_green = final_scores.mean()
            
            summary = None
            if self.return_summary:
                summary = self.model.module.compute_summary(mean_green, final_scores.std(), responses)
                
            return mean_green, final_scores, responses, summary


if __name__ == '__main__':
    # from green_score import GREEN
    import time

    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN-radllama2-7b",
        do_sample=False,  # should be always False
        batch_size=16,
        return_0_if_no_green_score=True,
        cuda=True,
        return_summary=True
    )
    t = time.time()
    refs = [
               "Interstitial opacities without changes.",
               "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
               "Bibasilar atelectasis. Otherwise, no acute intrathoracic process.",
               "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
               "Interval resolution of previously seen mild pulmonary edema with trace bilateral pleural effusions.",
               "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
               "Bilateral pleural effusions, large on the right and small on the left. No definite focal consolidation identified, although evaluation is limited secondary to these effusions.",
               "1. Mild left basal atelectasis. Otherwise unremarkable. 2. No definite displaced rib fracture though if there is continued concern dedicated rib series may be performed to further assess.",
           ] * 1
    hyps = [
               "Interstitial opacities at bases without changes.",
               "Interval resolution of previously seen mild pulmonary edema with trace bilateral pleural effusions.",
               "Bibasilar atelectasis. Otherwise, no acute intrathoracic process.",
               "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
               "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
               "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
               "In comparison with the study of ___, the increased opacification at the right base has essentially cleared with better inspiration. Cardiac silhouette remains at the upper limits of normal in size and there is again tortuosity of the aorta without vascular congestion or pleural effusion. Biapical changes, especially on the right, are stable.",
               "1. Mild left basal atelectasis. Otherwise unremarkable.",
           ] * 1

    mean_green, greens, text, summary = model(refs=refs, hyps=hyps)
    print("Mean reward for the given examples is: ", mean_green)
    print("Array of rewards for the given examples is: ", greens)
    print(summary)
    print(text[0])
    print(time.time() - t)
    print(len(hyps))
