import re
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pandas as pd
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import os
from tqdm import tqdm
import numpy as np
import time
from green_score.utils import (
    gather_processes,
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)
import sys
import warnings

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def tqdm_on_main(*args, **kwargs):
    if is_main_process():
        print("==== Beginning Inference ====")
        return tqdm(*args, **kwargs)
    else:
        return kwargs.get('iterable', None)

class Inferer:
    def __init__(
        self,
        dataset=None,
        model=None,
        tokenizer=None,
        model_name="",
        output_dir=".",
        num_examples=None,
        batch_size=10,
        max_length=2048,
    ):

        self.dataset = Dataset.from_dict(
            {"reference": dataset[0], "prediction": dataset[1]}
        )
        self.process_data()

        self.model = model
        self.model_name = model_name.split("/")[-1] 
        self.tokenizer = tokenizer
        self.num_examples = num_examples

        self.output_dir = output_dir

        self.batch_size = batch_size

        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

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

        self.max_length = max_length

    def process_data(self):
        print("Processing data...making prompts")

        def promting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }

        self.dataset = self.dataset.map(promting, batched=True)
        print("Done.")

    @torch.inference_mode()
    def infer(self):

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            dataset_dist = split_dataset_by_node(
                self.dataset,
                rank=get_rank(),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            print("Distributed dataset created on rank: ", int(os.environ["RANK"]))
        else:
            dataset_dist = self.dataset

        local_completions = []
        local_references = []

        for batch in tqdm_on_main(
            iterable=dataset_dist.iter(batch_size=self.batch_size),
            total=len(dataset_dist) // self.batch_size,
        ):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

        # gather results if multi gpu and single gpu settings
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.completions, self.prompts = gather_processes(
                local_completions, local_references
            )
        else:
            self.completions = local_completions
            self.prompts = local_references

        if is_main_process():
            print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("length of prompts and completions are not equal!")

        self.process_results()

    def tokenize_batch_as_chat(self, batch):
        print(batch)
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        # tokenization
        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(int(os.environ.get("LOCAL_RANK", 0)))

        return batch

    def get_response(self, batch):

        # format batch
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        # batch["conv"] = [
        #     [
        #         {"from": "human", "value": i},
        #     ]
        #     for i in batch["prompt"]
        # ]
        
        batch = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch["prompt"]]

        batch = self.tokenize_batch_as_chat(batch)

        outputs = self.model.generate(
            # **batch,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        # # decode response
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # reformat the responses
        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def process_results(self):

        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "evaluation": self.completions,
                "green": self.green_scores,
                **self.error_counts,  # unpacking the dictionary
            }
        )
        path = self.output_dir + f"/results_{self.model_name}.csv"
        os.makedirs(self.output_dir, exist_ok=True)
        print("Saving generated response to prompt to ", path)
        results_df.to_csv(path, index=False)

        self.compute_summary()

        return results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        # matched findings, we want to look at the sum of all errors
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        # significant clinical errors, we want to look at each error type
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        # matched findings, we want to look at the sum of all errors
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        # set the prior study (sub_categories: (e) Mentioning a comparison that isn't in the reference, (f) Omitting a comparison detailing a change from a prior study) errors to 0
        # Note: we are NOT doing this anymore: sig_errors[-2:] = 0, 0

        if matched_findings == 0:
            return 0

        if (
            sig_present is None or matched_findings is None
        ):  # when the template does not include the key "Clinically Significant Errors"
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):

        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        # Pattern to match integers within the category, stopping at the next category or end of text
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        # Initialize the counts
        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        # If the category is not found, return 0
        if not category_text:
            if for_reward:
                # we need to know whether the category is empty or not, otherwise we overesitmate the reward
                return None, None
            return sum_counts, sub_counts
        # If the category is found, but the category is empty, return 0
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        # Possible fine-grained error categories for categories Significant and Insignificant Clinical Errors
        else:  # "Clinically Significant Errors" or "Clinically Insignificant Errors"
            # Split each string at the first space and keep only the first part
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            # Find all sub_categories in the matched text
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            # this is for the gpt-4 template which assigns a number to the subcategories not letters
            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                # need to loop over all matches, because the sub_categories are not always in the same order
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        # If the sub_category is found, insert the count to sub_counts at the ordered position
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            # take the first number after the colon
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

    def compute_sentences(self, response):
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

    def compute_summary(self):
        """
        Makes green summary.

        Args:
            mean_green (int): grean average.
            mean_std (int): grean std.
            responses (list): list of green model responses (str)

        Returns:
            str: green summary.
        """
        print("Computing summary ...")
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)

        summary = f"\n-------------{self.model_name}----------------\n [Summary]: Green average {np.mean(self.green_scores)} and standard variation {np.std(self.green_scores)} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n (a) False report of a finding in the candidate: {accuracies[self.sub_categories[0]]}. \n {representative_sentences[self.sub_categories[0]]} \n\n (b) Missing a finding present in the reference: {accuracies[self.sub_categories[1]]}. \n {representative_sentences[self.sub_categories[1]]} \n\n (c) Misidentification of a finding's anatomic location/position: {accuracies[self.sub_categories[2]]}. \n {representative_sentences[self.sub_categories[2]]} \n\n (d) Misassessment of the severity of a finding: {accuracies[self.sub_categories[3]]}. \n {representative_sentences[self.sub_categories[3]]} \n\n (e) Mentioning a comparison that isn't in the reference: {accuracies[self.sub_categories[4]]}. \n {representative_sentences[self.sub_categories[4]]} \n\n (f) Omitting a comparison detailing a change from a prior study: {accuracies[self.sub_categories[5]]}. {representative_sentences[self.sub_categories[5]]}.\n----------------------------------\n"

        print(summary)


def GREEN(model_name, refs, hyps, output_dir="."):
    warnings.filterwarnings("ignore", message="A decoder-only architecture is being used*") # this warning appears, despide 'padding_side='left' and correct padding
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning, message="Number of distinct clusters.*") # test examples are copied
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

    
    chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
            )  # 'nccl' is recommended for GPUs
            torch.cuda.set_device(dist.get_rank())
            if dist.get_rank() == 0:
                print("Distributed training with", torch.cuda.device_count(), "GPUs")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False if "Phi" in model_name else True,
        device_map={"": "cuda:{}".format(torch.cuda.current_device())},
        torch_dtype=torch.float16,
    )
        
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    
    tokenizer.chat_template = chat_template
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = True
    tokenizer.padding_side= "left"

    inferer = Inferer(
        dataset=[refs, hyps],
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        output_dir=output_dir,
        batch_size=4,
    )

    t = time.time()

    inferer.infer()

    t = time.time() - t
    print("Seconds per example: ", t / len(refs))
    
    if not is_main_process():
        # Exit the process
        print(f"Rank {dist.get_rank()} exiting.")
        dist.destroy_process_group()  # Clean up the distributed processing group
        sys.exit()  # Exit the process

if __name__ == "__main__":
    import time

    refs = [
        "[Breathing: Lungs] **Low lung volumes are noted.** [Breathing: Lungs] **There is diffuse prominence of the interstitium with indistinct pulmonary vascular markings.**",
        "[Breathing: Lungs] **Low lung volumes are noted.** [Breathing: Lungs] **There is diffuse prominence of the interstitium with indistinct pulmonary vascular markings.**",
        "[Breathing: Lungs] **Low lung volumes are noted.** [Breathing: Lungs] **There is diffuse prominence of the interstitium with indistinct pulmonary vascular markings.**",
        "[Breathing: Lungs] **Low lung volumes are noted.** [Breathing: Lungs] **There is diffuse prominence of the interstitium with indistinct pulmonary vascular markings.**",
        
    ]
    hyps = [
        "[Breathing Lungs] There are low lung volumes, which limit the evaluation of the lung bases. [Breathing Lungs] **There is a diffuse reticular pattern seen throughout the lungs, which may represent pulmonary edema, atypical infection or a chronic interstitial process.** [Cardiac Heart Size] The cardiomediastinal silhouette is stable. [Breathing Lungs] No focal consolidation is seen. [Breathing Pleura] There are no pleural effusions. [Everything else Bones] There are no acute osseous abnormalities. [Everything else Bones] Degenerative changes of the spine are present.",
        "[Breathing Lungs] There are low lung volumes, which limit the evaluation of the lung bases. [Breathing Lungs] **There is a diffuse reticular pattern seen throughout the lungs, which may represent pulmonary edema, atypical infection or a chronic interstitial process.** [Cardiac Heart Size] The cardiomediastinal silhouette is stable. [Breathing Lungs] No focal consolidation is seen. [Breathing Pleura] There are no pleural effusions. [Everything else Bones] There are no acute osseous abnormalities. [Everything else Bones] Degenerative changes of the spine are present.",
        "[Breathing Lungs] There are low lung volumes, which limit the evaluation of the lung bases. [Breathing Lungs] **There is a diffuse reticular pattern seen throughout the lungs, which may represent pulmonary edema, atypical infection or a chronic interstitial process.** [Cardiac Heart Size] The cardiomediastinal silhouette is stable. [Breathing Lungs] No focal consolidation is seen. [Breathing Pleura] There are no pleural effusions. [Everything else Bones] There are no acute osseous abnormalities. [Everything else Bones] Degenerative changes of the spine are present.",
        "[Breathing Lungs] There are low lung volumes, which limit the evaluation of the lung bases. [Breathing Lungs] **There is a diffuse reticular pattern seen throughout the lungs, which may represent pulmonary edema, atypical infection or a chronic interstitial process.** [Cardiac Heart Size] The cardiomediastinal silhouette is stable. [Breathing Lungs] No focal consolidation is seen. [Breathing Pleura] There are no pleural effusions. [Everything else Bones] There are no acute osseous abnormalities. [Everything else Bones] Degenerative changes of the spine are present."
    ]

    model_name = "StanfordAIMI/GREEN-radllama2-7b"

    compute(model_name, refs, hyps, output_dir=".")
