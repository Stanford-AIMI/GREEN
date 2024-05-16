def process_responses(responses):
    """
    Processes a list of responses by removing unwanted tokens and returns the cleaned responses.

    Args:
        responses (list): List of response strings.

    Returns:
        list: List of cleaned response strings.
    """
    response_list = []
    for i in responses:
        if "" in i:
            i = i.split("")[-1]
        i = i.replace("</s>", "").replace("<unk>", "")
        response_list.append(i)
    return response_list


def make_prompt(text1, text2):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


def tokenize_batch_as_chat(tokenizer, batch):
    """
    Tokenizes a batch of prompts as chat input for the model.

    Args:
        tokenizer: Tokenizer object for encoding the batch.
        batch (list): List of prompts to be tokenized.

    Returns:
        dict: Tokenized batch with input IDs and attention masks.
    """
    batch = [
        tokenizer.apply_chat_template(
            i, tokenize=False, add_generation_prompt=True
        )
        for i in batch
    ]

    batch = tokenizer.batch_encode_plus(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    return batch
