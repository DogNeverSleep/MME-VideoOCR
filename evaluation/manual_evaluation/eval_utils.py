import json


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def get_prompt(item):
    eval_method = item["eval_method"]
    prompt = ""
    question = item["question"]
    if eval_method == "containment_match":
        recognition_prompt = "Based on the video and the question below, directly answer the content that needs to be recognized in plain text. Do not include any additional explanations, formatting changes, or extra information."
        post_prompt = "The answer is:"
        prompt = (
            recognition_prompt + "\n" + "Question: " + question + "\n" + post_prompt
        )
    elif eval_method == "multiple_choice":
        option = item["option"]  # list
        multiple_chocie_prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option."
        option_prompt = "Option:\n"
        for i, c in enumerate(option):
            option_prompt += f"{chr(65 + i)}. {c}\n"
        post_prompt = "The best answer is:"
        prompt = (
            multiple_chocie_prompt
            + "\n"
            + "Question: "
            + question
            + "\n"
            + option_prompt
            + post_prompt
        )
    elif eval_method == "gpt_assisted_scoring":
        pre_prompt = "Based on the video and the question below, directly provide the answer in plain text. Do not include any additional explanations, formatting changes, or extra information."
        post_prompt = "The answer is:"
        prompt = pre_prompt + "\n" + "Question: " + question + "\n" + post_prompt
    return prompt


if __name__ == "__main__":
    dataset_json_file = ""
    dataset = load_json(dataset_json_file)
    for item in dataset:
        print(get_prompt(item))
