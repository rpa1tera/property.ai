import json

from src.evaluation.intent_eval import evaluate_intents


def _write_golden(tmp_path, items):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_evaluate_all_correct(tmp_path):
    items = [
        {"question": "O que é franquia?", "intent_expected": "franquia"},
        {"question": "Quero falar com um atendente", "intent_expected": "escalonamento"},
        {"question": "Meu seguro cobre incêndio?", "intent_expected": "cobertura"},
    ]
    path = _write_golden(tmp_path, items)
    result = evaluate_intents(path)
    assert result["total"] == 3
    assert result["correct"] == 3
    assert result["accuracy"] == 1.0
    assert result["errors"] == []


def test_evaluate_detects_error(tmp_path):
    items = [
        {"question": "Olá bom dia", "intent_expected": "cobertura"},
    ]
    path = _write_golden(tmp_path, items)
    result = evaluate_intents(path)
    assert result["accuracy"] == 0.0
    assert len(result["errors"]) == 1
    assert result["errors"][0]["expected"] == "cobertura"


def test_golden_set_real_accuracy_above_threshold():
    """Acurácia mínima de 95% no golden_set.json real."""
    result = evaluate_intents("data/evaluation/golden_set.json")
    assert result["accuracy"] >= 0.95, (
        f"Acurácia caiu para {result['accuracy']:.1%}. Erros: {result['errors']}"
    )
