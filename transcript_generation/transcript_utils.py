import openai
import json
import tqdm
import random
import pandas as pd
import argparse

import os


def get_secret_key():
    # Load secret API key
    with open('openai.key', 'r') as f:
        openai.api_key = f.readline()


def get_args():
    parser = argparse.ArgumentParser(
        description=
        """
        A pipeline for generating transcripts using FAQ prompts and GPT3
        """)

    parser.add_argument("--query_path",
                        help="Path to faq or squad queries.",
                        action="store",
                        dest="query_path",
                        required=True, default=None)
    parser.add_argument("--output_path",
                        help="Directory to store the generated transcripts.",
                        action="store",
                        dest="output_path",
                        required=True, default=None)
    parser.add_argument("--kb_name",
                        help="Name of kb (used for Agent greeting prompt).",
                        action="store",
                        dest="kb_name",
                        required=False, default='Walmart')
    args = parser.parse_args()
    return args


def save_transcript(output_path, i, transcript):
    fname = os.path.join(output_path, 'file_{}.txt'.format(i))
    with open(fname, 'w') as f:
        f.write(transcript)


def load_squad_queries(squad_path):
    queries = []
    answers = []
    with open(squad_path, 'r') as f:
        data = json.load(f)
    for doc in data['data']:
        for paragraph in doc['paragraphs']:

            for qa in paragraph['qas']:
                query = qa['question']
                answer = qa['answers'][0]['text']
                queries.append(query)
                answers.append(answer)
    return queries, answers


def load_faqs(faq_path):
    faqs = pd.read_csv(faq_path)
    questions = list(faqs.question)
    answers = list(faqs.answer)
    return questions, answers