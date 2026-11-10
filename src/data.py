"""Read training questions from pinned Hub datasets or local JSONL."""
from itertools import islice
import json
from pathlib import Path

from .rollout import answer_key, boxed_answer


HUB_DATASETS = {
    "math": ("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"),
    "numina": ("AI-MO/NuminaMath-1.5", "1b05109f9e5c1ad06c0663519502416c30b300f8"),
    "deepmath": ("zwhe99/DeepMath-103K", "5cf055d1fe3d7a2eb19719ac020211469736ae44"),
}
MATH_SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
MISSING_ANSWERS = {"proof", "notfound", "none", "nan", "n/a"}


def hub_rows(source):
    from datasets import load_dataset

    repo, revision = HUB_DATASETS[source]
    options = dict(revision=revision, split="train", streaming=True)
    subjects = MATH_SUBJECTS if source == "math" else (None,)
    for subject in subjects:
        for row in load_dataset(repo, subject, **options):
            if source == "math":
                question = row["problem"]
                answer = boxed_answer(row["solution"] or "")
            elif source == "numina":
                if (row["question_type"] != "math-word-problem" or row["problem_is_valid"] != "Yes"
                        or row["solution_is_valid"] != "Yes"):
                    continue
                question, answer = row["problem"], row["answer"]
            else:
                question, answer = row["question"], row["final_answer"]
            question = (question or "").strip()
            answer = (answer or "").strip()
            if answer.startswith(r"\boxed{"):
                answer = boxed_answer(answer) or ""
            if question and answer_key(answer) and answer.casefold() not in MISSING_ANSWERS:
                yield {"question": question, "answer": answer}


def jsonl_rows(path, with_answers):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question, answer = row.get("question"), row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{number} needs a nonempty question")
            if with_answers and (not isinstance(answer, str) or not answer_key(answer)):
                raise ValueError(f"{path}:{number} needs a bare final answer")
            yield row


def read_questions(dataset, with_answers, limit=None):
    source = str(dataset)
    rows = hub_rows(source) if source in HUB_DATASETS else jsonl_rows(Path(source), with_answers)
    questions = list(islice(rows, limit))
    if not questions:
        raise ValueError(f"No training questions in {source}")
    return questions
