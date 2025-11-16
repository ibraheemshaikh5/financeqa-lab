import os
import json
from typing import Tuple

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# configurations
N_EXAMPLES = 5 # number of example questions

TARGET_MODEL = "gpt-4o-mini"    # model that's evaluated
LABEL_MODEL = "gpt-4o"          # stronger model that's labeling the failures

OUT_CSV = "data/financeqa_labeled.csv"

# the potential failure classifications for this model 
ERROR_LABELS = [
    "CORRECT",
    "ARITHMETIC_ERROR",
    "ACCOUNTING_CONVENTION_ERROR",
    "MISSING_OR_WRONG_ASSUMPTION",
    "WRONG_METRIC_OR_CONCEPT",
    "CONTEXT_MISUSE_OR_HALLUCINATION",
    "NON_ANSWER_OR_GENERIC",
]

# system prompts for labeling
SYSTEM_LABELING = f"""
You are a senior buy-side financial analyst.

Given:
1) a FinanceQA question,
2) the correct answer (truth),
3) a model's answer,

decide whether the model's answer is correct. If it is wrong, assign
the SINGLE PRIMARY failure type from this list:

{ERROR_LABELS}

Definitions:
- ARITHMETIC_ERROR: Numbers from the context are used but math is wrong.
- ACCOUNTING_CONVENTION_ERROR: Violates standard accounting practice
  (e.g., mixing basic/diluted shares, pre/post-tax, GAAP vs non-GAAP).
- MISSING_OR_WRONG_ASSUMPTION: Fails because the model makes bad
  assumptions or misses required assumptions.
- WRONG_METRIC_OR_CONCEPT: Confuses metrics (EBITDA vs operating income,
  cash vs accrual, margin vs absolute dollars, etc.).
- CONTEXT_MISUSE_OR_HALLUCINATION: Ignores given context or invents
  line items/values not in the document.
- NON_ANSWER_OR_GENERIC: Hand-wavy commentary, refuses, or doesn't
  actually answer the question.
- CORRECT: The model's answer matches the truth and is consistent
  with finance conventions.

Return STRICT JSON only in the form:
{{"label": "...", "rationale": "..."}}
""".strip()

# function to get the API client from openAI
def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY needs to be set in the enviornment")
    return OpenAI(api_key=api_key)

# use Hugging Face to load a random sample of FinanceQA questions
# uses basic and assumption questions for 'n' rows
def load_financeqa_sample(n: int) -> pd.DataFrame:
    ds = load_dataset("AfterQuery/FinanceQA", split="test")
    df = ds.to_pandas()

    # use a mask to get all of the questions from the dataset (basic + assumption)
    if "question_type" in df.columns:
        mask = df["question_type"].isin(["basic", "assumption"])
        df = df[mask]

    # restrict for number of questions requested
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return df.sample(n=n, random_state=42).reset_index(drop=True)

# call the weaker model to answer the FinanceQA question
def call_target_model(client: OpenAI, question: str, context:str) -> str:
    prompt = f"""
    You are a professional equity research analyst.

    Use the context if helpful, but keep the answer concise and numeric-first.

    Context:
    {context}

    Question:
    {question}

    Answer with the final numeric answer first, then one short sentence of explanation.
    """
    
    resp = client.chat.completions.create(
        model=TARGET_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return resp.choices[0].message.content.strip()

# ask the stronger model to classify the failure
def label_failure(client: OpenAI, question: str, truth: str, model_answer: str):
    user_msg = f"""
        Question: 
        {question}

        Actual answer:
        {truth}

        Model Answer:
        {model_answer}

        Is the model's answer correct? If not, choose ONE label from:
        {ERROR_LABELS}
        
        Return JSON {{"label": "[Your Answer Here]", "rationale": "[Your Answer Here]"}}
    """

    # the JSON schema for a structured output
    labeling_schema = {
        "name": "finance_label_schema",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "rationale": {"type": "string"}
            },
            "required": ["label", "rationale"],
            "additionalProperties": False
        },
        "strict": True
    }

    # ensure that the data is outputted in json
    try:
        resp = client.chat.completions.create(
            model=LABEL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_LABELING},
                {"role": "user", "content": user_msg}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": labeling_schema
            },
            temperature=0,
        )

        # accessed parsed content
        content = resp.choices[0].message.content

        if content:
            data = json.loads(content)
            label = data.get("label", "UNKNOWN")
            rationale = data.get("rationale", "")
        else:
            label = "UNKNOWN"
            rationale = "No content returned in JSON"
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        label = "UNKNOWN"
        rationale = f"JSON decode error: {str(e)}"
    except Exception as e:
        print(f"Error in label_failure: {e}")
        label = "UNKNOWN"
        rationale = str(e)
    
    if label not in ERROR_LABELS:
        print(f"Warning: Invalid label: '{label}' returned. Storing as UNKNOWN.")
        label = "UNKNOWN"

    return label, rationale

def main():
    os.makedirs("data", exist_ok=True)
    client = get_client()

    df = load_financeqa_sample(N_EXAMPLES)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Labeling failures"):
        question = row["question"]
        answer = row["answer"]
        context = row.get("context", "")

        # first get the lower model's answer
        try:
            model_answer = call_target_model(client, question, context)
        except Exception as e:
            print("Error calling TARGET_MODEL:", e)
            model_answer = ""

        # now get the failure label from the strong model
        try:
            label, rationale = label_failure(client, question, answer, model_answer)
        except Exception as e:
            print("Error calling LABEL_MODEL:", e)
            label, rationale = "UNKNOWN", str(e)

        record = {
            "question": question,
            "answer": answer,
            "context": context,
            "question_type": row.get("question_type", ""),
            "company": row.get("company", ""),
            "file_link": row.get("file_link", ""),
            "file_name": row.get("file_name", ""),
            "model_answer": model_answer,
            "error_label": label,
            "error_rationale": rationale,
        }

        records.append(record)

    # save as a csv
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV, index=False)
    
    # log in console that it's saved as a csv
    print(f"The labeled data has been saved to {OUT_CSV}")
    print(out_df["error_label"].value_counts())

if __name__ == "__main__":
    main()
