import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2
import numpy as np
import json
import requests
import time
from openai import OpenAI
import contextlib


MODEL_VERSION = "gpt-4o-2024-08-06"
API_KEY = ""
BASE_URL = ""
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved JSON to {file_path}")


def get_chat_response(
    prompt: str,
    sys_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    retries: int = 10,
):
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": sys_prompt,
        },
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if attempt == retries - 1:
                return ""
        except Exception as e:
            print(f"Error: {e}")
            return ""


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer",
        "Answer is",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""

    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


GPT_PROMPT = """You are a professional bilingual translation evaluator.
Here are two sentences: one in Chinese and one in English.

Sentence 1: SENTENCE_1
Sentence 2: SENTENCE_2

Please evaluate whether the two sentences convey the same meaning and can be considered accurate translations of each other.

If the meanings are equivalent and the translation is accurate, respond with "correct".
If there are significant differences in meaning or inaccuracies in translation, respond with "wrong".

You must only respond with one word: "correct" or "wrong". Do not provide any explanations, comments, or additional text.
Focus solely on semantic equivalence, not grammar or style. Ignore minor differences as long as the meaning is preserved."""


def process_result_of_single_item(item):
    pred = item["model_response"]
    if pred is None:
        score = 0.0
    else:
        metric = item["eval_method"].strip()
        ground_truth = item["answer"].strip()
        if metric == "containment_match":
            if (
                item["task"] == "trajectory_recognition"
                or item["task"] == "scrambled_recognition"
            ):
                if pred == ground_truth:
                    score = 1.0
                else:
                    score = 0.0
            else:
                ground_truth.replace("’", "'").lower()
                pred = pred.replace("’", "'").lower()
                if ";" in ground_truth:
                    answer_list = ground_truth.split(";")
                    answer_list = [ans.strip() for ans in answer_list]
                    answer_list = [ans.replace("’", "'") for ans in answer_list]
                    for ans in answer_list:
                        if ans not in pred:
                            print(f"ans: {ans} not in pred: {pred}")
                            score = 0.0
                            break
                    else:
                        score = 1.0
                else:
                    if ground_truth in pred:
                        score = 1.0
                    else:
                        score = 0.0
        elif metric == "multiple_choice":
            pred_ans = extract_characters_regex(pred)
            if pred_ans == ground_truth:
                score = 1.0
            else:
                score = 0.0
        elif metric == "gpt_assisted_scoring":
            gpt_prompt = GPT_PROMPT
            gpt_prompt = gpt_prompt.replace("SENTENCE_1", ground_truth)
            gpt_prompt = gpt_prompt.replace("SENTENCE_2", pred)
            score = -1
            try_num = 0
            while score == -1 and try_num <= 10:
                try:
                    response = get_chat_response(prompt=gpt_prompt)
                    if "correct" in response.lower():
                        score = 1.0
                    elif "wrong" in response.lower():
                        score = 0.0
                    else:
                        score = -1
                        try_num += 1
                except Exception as e:
                    print(f"Error: {e}")
                    print("Retrying...\n")
            if score == -1:
                print(f"GPT Error\nresponse: {response}")
                score = 0.0
    return item["task_type"], item["task"], score


def process_model_response(input_file, model_name, res_output_path):
    data = load_json(input_file)

    task_type_num = {}
    task_num = {}
    task_type_score = {}
    task_score = {}
    sum_score = 0

    for item in data:
        task_type, task, score = process_result_of_single_item(item)
        task_type_num[task_type] = task_type_num.get(task_type, 0) + 1
        task_num[task] = task_num.get(task, 0) + 1
        task_type_score[task_type] = task_type_score.get(task_type, 0) + score
        task_score[task] = task_score.get(task, 0) + score
        sum_score += score

    res_txt = os.path.join(
        res_output_path,
        model_name + ".txt",
    )
    with open(res_txt, "w") as f:
        with contextlib.redirect_stdout(f):
            # task type accuracy
            print("Task Type Accuracy")
            for k, v in task_type_num.items():
                print(f"{k}: {task_type_score[k]} / {v}")
                print(f"{k}: {task_type_score[k] / v}")
                print()
            # task accuracy
            print("-" * 50)
            print("Task Accuracy")
            for k, v in task_num.items():
                print(f"{k}: {task_score[k]} / {v}")
                print(f"{k}: {task_score[k] / v}")
                print()
            # overall accuracy
            print("-" * 50)
            print("Overall Accuracy")
            print(f"Overall: {sum_score} / {2000}")
            print(f"Overall Accuracy: {sum_score / 2000}")


if __name__ == "__main__":
    input_file = ""
    model_name = ""
    res_output_path = ""
    process_model_response(input_file, model_name, res_output_path)
