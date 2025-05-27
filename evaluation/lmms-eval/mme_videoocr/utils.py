import datetime
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import json
import requests
from openai import OpenAI
from loguru import logger as eval_logger

# NOTICE
LOCAL_VIDEO_PATH = ""

MODEL_VERSION = "gpt-4o-2024-08-06"
API_KEY = ""
BASE_URL = ""
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


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


def mme_videoocr_doc_to_visual(doc):
    video = os.path.join(LOCAL_VIDEO_PATH, doc["video_index"] + ".mp4")
    return video


def mme_videoocr_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = ""
    question = doc["question"].strip()
    metric = doc["eval_method"].strip()
    if metric == "containment_match":
        recognition_prompt = "Based on the video and the question below, directly answer the content that needs to be recognized in plain text. Do not include any additional explanations, formatting changes, or extra information."
        post_prompt = "The answer is:"
        prompt = (
            recognition_prompt + "\n" + "Question: " + question + "\n" + post_prompt
        )
    elif metric == "multiple_choice":
        option = doc["option"]  # list
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
    elif metric == "gpt_assisted_scoring":
        pre_prompt = "Based on the video and the question below, directly provide the answer in plain text. Do not include any additional explanations, formatting changes, or extra information."
        post_prompt = "The answer is:"
        prompt = pre_prompt + "\n" + "Question: " + question + "\n" + post_prompt

    return prompt


def mme_videoocr_process_results(doc, results):
    pred = results[0]
    metric = doc["eval_method"].strip()
    ground_truth = doc["answer"].strip()
    if metric == "containment_match":
        task = doc["task"]
        if task == "trajectory_recognition" or task == "scrambled_recognition":
            if pred == ground_truth:
                score = 1.0
            else:
                score = 0.0
        else:
            ground_truth = ground_truth.replace("’", "'").lower()
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
        print(f"pred_ans: {pred_ans}")
        if pred_ans == ground_truth:
            score = 1.0
        else:
            score = 0.0
    elif metric == "gpt_assisted_scoring":
        task = doc["task"]
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
            print(f"GPT Error")
            score = 0.0
    data_dict = {
        "task_type": doc["task_type"],
        "task": doc["task"],
        "metric": metric,
        "score": score,
    }
    return {"score": data_dict}


def mme_videoocr_aggregate_results(results):
    task_type_scores = {}
    task_scores = {}
    metric_scores = {}
    metric_num = {}
    total_score = 0.0
    task_num = {}
    task_type_num = {}
    for res in results:
        task_type = res["task_type"]
        task = res["task"]
        score = res["score"]
        metric = res["metric"]
        task_num[task] = task_num.get(task, 0) + 1
        task_type_num[task_type] = task_type_num.get(task_type, 0) + 1
        metric_num[metric] = metric_num.get(metric, 0) + 1
        metric_scores[metric] = metric_scores.get(metric, 0) + score
        if task_type not in task_type_scores:
            task_type_scores[task_type] = score
        else:
            task_type_scores[task_type] += score
        if task not in task_scores:
            task_scores[task] = score
        else:
            task_scores[task] += score
        total_score += score

    print("\n")
    print("-" * 50)
    print(f"Task Type Accuracy:")
    for key, value in task_type_scores.items():
        print(f"{key}: {value}")
        accuracy = value / task_type_num[key]
        print(f"{key}: {accuracy}")
        print()
    print("-" * 50)
    print(f"Task Accuracy:")
    for key, value in task_scores.items():
        print(f"{key}: {value} / {task_num[key]}")
        # accuracy
        accuracy = value / task_num[key]
        print(f"{key}: {accuracy}")
        print()
    print("-" * 50)
    print("Overall Accuracy:")
    print(f"{total_score} / 2000")
    print("Accuracy: {}".format(total_score / 2000.0))
    print("-" * 50)
    print("\n")

    return total_score / 2000
