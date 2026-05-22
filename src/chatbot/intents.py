import re

_INTENT_PATTERNS: dict[str, list[str]] = {
    "escalonamento": [
        r"falar com (um |uma )?(atendente|humano|pessoa|especialista|gerente)",
        r"quero atendimento humano",
        r"me (transfere|conecta|passa) (para|a) (um |uma )?(atendente|pessoa|humano)",
        r"passa (para|pra) (um |uma )?(humano|atendente|pessoa)",
        r"não consigo resolver",
        r"preciso de ajuda (humana|de uma pessoa)",
        r"fale (com|comigo).*(humano|atendente)",
    ],
    "cobertura": [
        r"cobr[ea]",
        r"\bcobertura\b",
        r"\bcobre\b",
        r"\bcoberto[as]?\b",
        r"\bcoberta[s]?\b",
        r"\binclui\b",
        r"está (coberto|coberta|incluído|garantido)",
        r"estão (cobertos|cobertas)",
        r"\bgarante\b",
        r"\brisco\b coberto",
        r"o que (está|fica) (incluso|coberto)",
        r"protege contra",
        r"\bindenizado[as]?\b",
        r"danos? .*(causados? |por )",
    ],
    "franquia": [
        r"\bfranquia\b",
        r"participação obrigatória",
        r"valor mínimo (do|de) (sinistro|dano)",
        r"desconto (da|na) indenização",
    ],
    "sinistro": [
        r"\bsinistro\b",
        r"\bacionar\b",
        r"\baciono\b",
        r"\bacidente\b",
        r"\bincêndio\b",
        r"\benchente\b",
        r"\binundação\b",
        r"\bprejuízo\b",
        r"como (abrir|abro|registr[ao]) (um |o )?(sinistro|ocorrência|boletim)",
        r"boletim de ocorrência",
        r"comunica[rç] (o |um )?sinistro",
        r"\bindeniza[çc][ãa]o\b",
        r"\bvistoria\b",
        r"recebo .*indeniza",
        r"recusad[ao]",
    ],
    "apolice_documentos": [
        r"\bapólice\b",
        r"\bapolice\b",
        r"segunda via",
        r"2[aª]\s*via",
        r"\bcertificado\b",
        r"número (da )?apólice",
        r"\bcontrato\b (de seguro)?",
        r"\brenovação\b",
        r"\brenovar\b",
        r"prazo (da |de )?vigência",
        r"\bvigência\b",
    ],
    "cancelamento": [
        r"\bcancelar\b",
        r"\bcancelamento\b",
        r"\bencerrar\b",
        r"\brescisão\b",
        r"\bdesistir\b",
        r"não quero mais",
        r"quero sair",
        r"suspender (o )?seguro",
    ],
}

_COMPILED: dict[str, list[re.Pattern]] = {
    intent: [re.compile(p, re.IGNORECASE) for p in patterns]
    for intent, patterns in _INTENT_PATTERNS.items()
}


def classify_intent(text: str) -> str:
    # escalonamento is checked first — explicit escalation always wins
    for intent in ("escalonamento", "cancelamento", "cobertura", "franquia", "sinistro", "apolice_documentos"):
        for pattern in _COMPILED[intent]:
            if pattern.search(text):
                return intent
    return "outros"
