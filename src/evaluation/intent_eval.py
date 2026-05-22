"""Avalia o classificador de intents contra o golden set.

Não usa LLM nem RAGAS — roda em <1s, ótimo para CI.

Uso:
    python -m src.evaluation.intent_eval
    python -m src.evaluation.intent_eval --golden data/evaluation/golden_set.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.chatbot.intents import classify_intent
from src.config import get_settings


def evaluate_intents(golden_path: str | Path) -> dict:
    """Roda classify_intent em cada item do golden set.

    Returns:
        dict com:
        - total: int
        - correct: int
        - accuracy: float (0..1)
        - confusion: {expected: {predicted: count}}
        - errors: lista de itens classificados errado
    """
    path = Path(golden_path)
    with path.open(encoding="utf-8") as f:
        items = json.load(f)

    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    errors: list[dict] = []

    for item in items:
        question = item["question"]
        expected = item.get("intent_expected", "outros")
        predicted = classify_intent(question)
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
        else:
            errors.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
            })

    total = len(items)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "errors": errors,
    }


def _format_report(result: dict) -> str:
    lines = [
        f"Total: {result['total']}",
        f"Acertos: {result['correct']}",
        f"Acurácia: {result['accuracy']:.1%}",
        "",
        "Matriz de confusão (linhas = esperado, colunas = previsto):",
    ]
    for expected, preds in sorted(result["confusion"].items()):
        preds_str = ", ".join(f"{p}={c}" for p, c in sorted(preds.items()))
        lines.append(f"  {expected:<20} -> {preds_str}")
    if result["errors"]:
        lines.append("")
        lines.append(f"Erros ({len(result['errors'])}):")
        for e in result["errors"]:
            lines.append(f"  [{e['expected']} -> {e['predicted']}] {e['question']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avalia o classificador de intents.")
    parser.add_argument(
        "--golden",
        default=None,
        help="Caminho do golden_set.json. Default: data/evaluation/golden_set.json.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.95,
        help="Acurácia mínima para passar (default 0.95). Falha com exit 1 se abaixo.",
    )
    args = parser.parse_args(argv)

    golden = args.golden or str(Path(get_settings().evaluation_dir) / "golden_set.json")
    result = evaluate_intents(golden)
    print(_format_report(result))

    return 0 if result["accuracy"] >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
