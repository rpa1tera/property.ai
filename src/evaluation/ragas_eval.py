import json
import os

from datasets import Dataset
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.chatbot.chain import answer
from src.rag.embeddings import get_embeddings, load_vectorstore


def _build_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    return LangchainEmbeddingsWrapper(get_embeddings())


def run_evaluation(golden_set_path: str | None = None) -> dict:
    golden_path = golden_set_path or "data/evaluation/golden_set.json"

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    vectorstore = load_vectorstore()
    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    questions, answers_list, contexts_list, ground_truths = [], [], [], []

    for item in golden:
        q = item["question"]
        gt = item["ground_truth"]

        result = answer(q, vectorstore=vectorstore)

        questions.append(q)
        answers_list.append(result["answer"])
        # RAGAS expects contexts as list of strings (actual chunk text)
        chunk_texts = [doc.page_content for doc in result.get("docs", [])]
        contexts_list.append(chunk_texts if chunk_texts else [""])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    scores = evaluate(dataset=dataset, metrics=metrics)
    return scores
