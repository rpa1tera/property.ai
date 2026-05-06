← [README](README.md) · [tasks](tasks.md)

# SPEC — InsurMinds: property.ai — ChatBot de Seguros Property

**Programa:** I2A2 — Atividade Obrigatória 2  
**Prazo de entrega:** 26/05/2026  
**Entrega para:** <challenges@i2a2.academy>  
**Assunto do e-mail:** `InsurMinds - Atividade obrigatória 2`  
**Chatbot pronto (para documentação):** 22/05/2026

---

## 1. Problema

Seguradoras do ramo **property** (seguro patrimonial) lidam com alto volume de atendimentos repetitivos — consultas sobre cobertura de imóveis, acionamento de sinistros, condições de apólice, franquias e documentação. O objetivo é automatizar esse atendimento com o **property.ai**, um ChatBot inteligente baseado em IA Generativa e RAG, liberando equipes para tarefas estratégicas.

---

## 2. Objetivo

Construir o **property.ai**, um ChatBot conversacional capaz de:

- Responder FAQs do segmento property com precisão e linguagem natural
- Recuperar informações de documentos (apólices, manuais, termos e condições, FAQs) via RAG
- Escalar para atendente humano quando necessário

**Fora de escopo:** processamento de sinistros em tempo real, integração com sistemas de pagamento, integração com CRM, acesso a tickets históricos, dados pessoais não anonimizados.

---

## 3. Glossário

| Termo | O que é |
|---|---|
| **RAG** | Técnica que faz o LLM responder com base em documentos recuperados, não só no que foi treinado |
| **Embedding** | Representação numérica (vetor) de um texto que captura seu significado semântico |
| **Vector Store** | Banco de dados especializado em buscar textos por similaridade semântica — usamos ChromaDB |
| **Chunking** | Quebrar documentos longos em pedaços menores para facilitar a recuperação precisa |
| **RAG-Fusion** | Variação do RAG que gera múltiplas versões da pergunta para ampliar a cobertura da busca |
| **CRAG Evaluator** | Componente que avalia se os chunks recuperados são suficientes antes de gerar a resposta |
| **LLM** | Modelo de linguagem grande — usamos `llama-3.3-70b-versatile` via Groq |
| **Golden Set** | Conjunto fixo de perguntas e respostas esperadas usado para medir a qualidade do RAG |

---

## 4. Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMADA DE INTERFACE                       │
│                    property.ai — Streamlit                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                      CAMADA DE ORQUESTRAÇÃO                      │
│            LangChain — Gerenciamento de contexto                 │
│              Memória de conversa (ConversationBufferMemory)       │
└──────────┬──────────────────────────────────────┬───────────────┘
           │                                       │
┌──────────▼───────────┐              ┌────────────▼──────────────┐
│      LLM ENGINE      │              │       RAG PIPELINE        │
│  Groq API (gratuito) │              │  1. RAG-Fusion            │
│  llama-3.3-70b-      │              │     gera 3 variações      │
│  versatile           │              │     semânticas da query   │
│  via ChatGroq        │              │  2. FAISS (top-k=5        │
│  (LangChain)         │              │     por variação)         │
│                      │              │  3. CRAG Evaluator        │
│                      │              │     avalia suficiência    │
│                      │              │  Embeddings:              │
│                      │              │  multilingual-e5-base     │
│                      │              │  Docs: Apólices, Manuais, │
│                      │              │  FAQs (PDF/HTML/CSV/JSON) │
└──────────────────────┘              └───────────────────────────┘
```

---

## 5. Stack Tecnológica

| Camada            | Tecnologia                                              | Notas                                        |
|-------------------|---------------------------------------------------------|----------------------------------------------|
| Linguagem         | Python 3.11+                                            |                                              |
| LLM               | **Groq API** — `llama-3.3-70b-versatile`                | Gratuito, ~6k tokens/min, ótimo em PT-BR     |
| Orquestração      | LangChain (`ChatGroq`)                                  | Integração nativa com Groq                   |
| Embeddings        | `intfloat/multilingual-e5-base` (HuggingFace)           | Treinado para retrieval, PT-BR nativo        |
| Vector Store      | **FAISS** (`faiss-cpu`, `langchain_community.FAISS`)    | `save_local`/`load_local`, sem servidor      |
| Interface         | Streamlit                                               |                                              |
| Ingestão de dados | PyMuPDF (PDFs), BSHTMLLoader, `langchain.document_loaders` |                                           |
| Avaliação RAG     | **RAGAS**                                               | Métricas automáticas sobre golden set        |
| Testes            | pytest                                                  |                                              |
| Documentação      | Markdown + draw.io (diagramas)                          |                                              |

### Por que `intfloat/multilingual-e5-base` para PT-BR?

- Treinado especificamente para **retrieval semântico** (tarefa exata do RAG), não apenas similaridade genérica
- Suporte nativo a 100 idiomas incluindo português brasileiro
- Melhor que `all-MiniLM-L6-v2` (inglês) e comparável a modelos maiores com metade do tamanho (~280MB)
- Roda 100% local, sem custo, via HuggingFace

### Por que Groq?

- Free tier: 14.400 requests/dia, sem cartão de crédito obrigatório
- `llama-3.3-70b-versatile` tem excelente desempenho em português
- Alternativa de fallback: `gemma2-9b-it` (mais leve, mesmo free tier)

### Por que FAISS em vez de ChromaDB?

ChromaDB foi a escolha original, mas a versão 1.5.9 causa `STATUS_ACCESS_VIOLATION` no Windows ao chamar `collection.add()` — o `onnxruntime 1.25.1` (dependência nativa do índice HNSW) não inicializa corretamente no ambiente de desenvolvimento. Tanto `PersistentClient` quanto `EphemeralClient` apresentam a falha.

FAISS (`faiss-cpu 1.13.2`) resolve o problema mantendo as mesmas garantias:

- Sem servidor, roda embedded
- Persistência via `vectorstore.save_local()` / `FAISS.load_local()` (dois arquivos: `faiss_index.faiss` + `faiss_index.pkl`)
- Interface idêntica no LangChain (`similarity_search`, `as_retriever`)
- Estável no Windows — amplamente testado em produção

---

## 6. Estratégia de Dados

### 6.1 Fontes

- **FAQs do segmento property** — coletadas e curadas por David Fagundes (perguntas/respostas em PT-BR)
- **Apólices property** — residencial, empresarial, condomínio
- **Manuais e Condições Gerais** — documentos públicos de seguradoras brasileiras

> **Formato do corpus ainda a confirmar por David Fagundes.** Suporte planejado: PDF, HTML, CSV e JSON.

### 6.2 Estrutura do Knowledge Base

```
data/
├── raw/
│   ├── faqs/
│   │   └── faq_property.csv         # perguntas/respostas curadas por David Fagundes
│   ├── apolices/                     # PDFs de apólices property
│   └── manuais/                      # condições gerais, termos
├── processed/
│   ├── chunks/                       # texto segmentado para RAG
│   └── embeddings/                   # vetores persistidos (ChromaDB)
└── evaluation/
    └── golden_set.json               # pares Q&A para avaliação RAGAS
```

### 6.3 Preparação

```
Extração (PyMuPDF / CSV loader)
       │
       ▼
  Limpeza / Tratamento  (cleaner.py)
       │
       ▼
  Chunking + Metadados
       │
       ▼
  Embedding + Indexação (ChromaDB)
```

**Etapa 1 — Extração (`corpus_loader.py`)**

`corpus_loader.py` detecta o formato pelo sufixo do arquivo e despacha para o loader adequado:

| Formato | Loader |
|---|---|
| `.pdf` | PyMuPDF (`fitz`) página a página |
| `.html` / `.htm` | `BSHTMLLoader` (LangChain + beautifulsoup4) |
| `.csv` | pandas — valida colunas `pergunta` e `resposta` |
| `.json` | `JSONLoader` (LangChain) |

**Etapa 2 — Limpeza (`cleaner.py`)**

| Problema | Tratamento |
|---|---|
| Cabeçalhos/rodapés repetidos (nº de página, nome da seguradora) | Regex para remover padrões fixos |
| Quebra de linha no meio de frase | Reune linhas que não terminam com `.`, `?`, `!` |
| Hifenização no fim de linha (`co-\nbertura`) | Regex: `(\w+)-\n(\w+)` → `\1\2` |
| Espaços múltiplos e caracteres de controle (`\x00`, ``) | `re.sub` + `strip()` |
| Encoding PT-BR quebrado | `ftfy` ou `chardet` + re-encode para UTF-8 |
| FAQs duplicadas | `pandas.drop_duplicates()` |
| Respostas muito curtas (< 10 palavras) | Filtro por tamanho mínimo |
| Boilerplate legal idêntico em todos os docs | Lista de strings a remover |

**Etapa 3 — Chunking por sentença + tamanho**

1. Quebrar o texto em sentenças por `[.!?]` (preserva integridade de cláusulas legais)
2. Agrupar sentenças respeitando `chunk_size=512` tokens e `overlap=64`

- Tag de metadado por tipo: `{"tipo": "faq" | "apolice" | "manual", "fonte": "<nome_arquivo>"}`

**Etapa 4 — Embedding + Indexação**

- Modelo: `intfloat/multilingual-e5-base`
- Persistência automática no ChromaDB (`persist_directory`)

---

## 7. RAG Pipeline — Fluxo Detalhado

```
Pergunta do usuário
       │
       ▼
  RAG-Fusion: LLM gera 3 variações semânticas da query
       │  (ex: "cobertura de dano" → "risco coberto", "indenização", "sinistro")
       ▼
  Busca semântica no ChromaDB para cada variação (top-k=5 cada)
  Deduplica e funde resultados → top-10 únicos
       │
       ▼
  CRAG Evaluator: LLM avalia se chunks são suficientes
       ├── SUFICIENTE → monta prompt e gera resposta
       └── INSUFICIENTE → escala para atendente humano
       │
       ▼
  Montagem do prompt estruturado com tags
       │
       ▼
  LLM (Groq) gera resposta fundamentada
       │
       ▼
  Resposta + fontes citadas → usuário
```

**Prompt System Template (estruturado com tags):**

```
Você é o property.ai, assistente especializado em seguros patrimoniais (property).
Responda APENAS com base nas informações dentro de <chunks>.
Se a resposta não estiver nos chunks, diga que vai encaminhar para um especialista.
Seja claro, educado e objetivo.

<chunks>
{context}
</chunks>

<query>
{question}
</query>

Resposta:
```

**Prompt RAG-Fusion (geração de variações):**

```
Gere 3 variações semânticas da pergunta abaixo para melhorar a busca em documentos
de seguros patrimoniais. Sem numeração, uma por linha, apenas as variações.

Pergunta: {question}
```

**Prompt CRAG Evaluator (avaliação de suficiência):**

```
Você é um avaliador de qualidade de recuperação de informação.
Avalie se os chunks abaixo contêm informação EXATA e SUFICIENTE para responder
à pergunta, sem depender de conhecimento externo.

<chunks>
{context}
</chunks>

<query>
{question}
</query>

Responda apenas: SUFICIENTE ou INSUFICIENTE
```

---

## 8. Avaliação — Golden Set e RAGAS

### 8.1 Golden Set

Conjunto fixo de **20–30 pares pergunta/resposta esperada** criados por David Fagundes, cobrindo os principais cenários do segmento property. Cada entrada inclui:

```json
{
  "question": "O que é franquia no seguro property?",
  "ground_truth": "Franquia é o valor mínimo de participação obrigatória do segurado...",
  "source_doc": "condicoes_gerais_residencial.pdf"
}
```

### 8.2 Métricas RAGAS

| Métrica | O que mede | Meta |
|---|---|---|
| **Faithfulness** | Resposta está fundamentada no contexto recuperado (sem alucinação) | ≥ 0.85 |
| **Answer Relevancy** | Resposta é relevante para a pergunta feita | ≥ 0.80 |
| **Context Precision** | Chunks recuperados são de fato úteis para a resposta | ≥ 0.75 |
| **Context Recall** | O contexto necessário foi recuperado (vs. ground truth) | ≥ 0.75 |

### 8.3 Fluxo de Avaliação

```
Golden Set (David Fagundes)
       │
       ▼
  Executa pipeline RAG para cada pergunta
       │
       ▼
  Coleta: pergunta + contexto recuperado + resposta gerada
       │
       ▼
  RAGAS calcula as 4 métricas automaticamente
       │
       ▼
  Relatório de avaliação → ajuste de prompts/chunking se necessário
```

---

## 9. Fluxos Conversacionais

### 9.1 Tipos de consulta mapeados — segmento property

| Intenção               | Exemplo de pergunta                                          |
|------------------------|--------------------------------------------------------------|
| Cobertura              | "Meu seguro cobre danos por enchente?"                       |
| Franquia               | "Qual é a franquia para sinistro de incêndio?"               |
| Sinistro               | "Como aciono o seguro após um roubo na empresa?"             |
| Apólice / Documentos   | "Como obter a 2ª via da minha apólice?"                      |
| Cancelamento           | "Quero cancelar meu seguro, como faço?"                      |
| Escalonamento          | "Quero falar com um atendente"                               |

### 9.2 Árvore de decisão simplificada

```
Entrada do usuário
    ├── Intenção identificada + resposta no KB → responde via RAG
    ├── Intenção identificada + sem resposta → resposta genérica + escala
    └── Intenção não identificada → pede reformulação (max 2x) → escala
```

---

## 10. Plano de Implementação

### Fase 1 — Fundação (01/05 – 08/05)

- [ ] Configurar repositório e ambiente virtual Python — **Raquel Alcantara**
- [ ] Coletar FAQs property e 5–8 PDFs de apólice/manual — **David Fagundes**
- [ ] Implementar pipeline de ingestão: `corpus_loader.py` (PDF/HTML/CSV/JSON) → limpeza → chunks → ChromaDB — **Millene Gomes**
- [ ] Validar recuperação semântica com queries de teste — **Millene Gomes**

### Fase 2 — Core ChatBot (09/05 – 13/05)

- [ ] Integrar LLM via Groq API (`ChatGroq`) — **Raquel Alcantara**
- [ ] Implementar `rag_fusion.py` — gera 3 variações semânticas da query e funde resultados — **Raquel Alcantara**
- [ ] Implementar `crag_evaluator.py` — avalia suficiência dos chunks antes de gerar resposta — **Raquel Alcantara**
- [ ] Construir `chain.py` com prompt estruturado `<chunks>/<query>` — **Raquel Alcantara**
- [ ] Implementar memória conversacional e detecção de intenções — **Raquel Alcantara**
- [ ] Cobrir os 6 fluxos de intenção mapeados — **Raquel Alcantara**
- [ ] Testes unitários das chains — **Raquel Alcantara**

### Fase 2.5 — Avaliação RAG (14/05 – 16/05, paralelo ao fim da Fase 2)

- [ ] Criar golden set com 20–30 pares Q&A property — **David Fagundes**
- [ ] Executar pipeline RAGAS sobre o golden set — **David Fagundes + Raquel Alcantara**
- [ ] Analisar métricas (Faithfulness, Answer Relevancy, Context Precision, Context Recall) — **David Fagundes + Raquel Alcantara**
- [ ] Ajustar chunking se métricas abaixo da meta — **Millene Gomes**
- [ ] Ajustar prompts se métricas abaixo da meta — **Raquel Alcantara**

### Fase 3 — Interface e Refinamento (17/05 – 21/05)

- [ ] Construir UI Streamlit com identidade property.ai — **Sueli da Hora Moreira**
- [ ] Adicionar lógica de escalonamento para humano — **Sueli da Hora Moreira**
- [ ] Exibir fontes dos documentos consultados — **Sueli da Hora Moreira**
- [ ] Testes manuais finais com David Fagundes validando respostas — **Sueli da Hora Moreira**
- [ ] Gerar evidências de execução (screenshots/video) — **Sueli da Hora Moreira**
- [ ] **Chatbot aprovado até 22/05**

### Fase 4 — Documentação e Entrega (22/05 – 25/05) — Fabricio Oliveira

- [ ] Escrever relatório PDF (estrutura abaixo)
- [ ] Gerar diagramas de arquitetura (draw.io)
- [ ] Incluir resultados RAGAS no relatório
- [ ] Organizar todos os artefatos
- [ ] Revisão final e envio em **26/05/2026**

---

## 11. Estrutura do Projeto

```
property.ai/                  # raiz do repositório git
├── data/
│   ├── raw/
│   │   ├── faqs/
│   │   ├── apolices/
│   │   └── manuais/
│   ├── processed/            # gerado — ignorado pelo .gitignore
│   │   ├── chunks/
│   │   └── embeddings/       # ChromaDB persist_directory
│   └── evaluation/
│       └── golden_set.json
├── src/
│   ├── ingestion/
│   │   ├── corpus_loader.py  # detecta formato (PDF/HTML/CSV/JSON) e extrai texto
│   │   ├── cleaner.py        # limpeza e tratamento do texto
│   │   └── chunker.py        # segmentação e overlap
│   ├── rag/
│   │   ├── embeddings.py     # criação/carregamento do vectorstore
│   │   └── retriever.py      # busca semântica
│   ├── chatbot/
│   │   ├── chain.py          # RAG chain principal
│   │   ├── rag_fusion.py     # geração de variações da query + fusão de resultados
│   │   ├── crag_evaluator.py # avaliador de suficiência dos chunks
│   │   ├── memory.py         # gerenciamento de histórico
│   │   └── intents.py        # classificação de intenções
│   ├── evaluation/
│   │   └── ragas_eval.py     # execução das métricas RAGAS
│   └── ui/
│       └── app.py            # interface Streamlit — property.ai
├── tests/
│   ├── test_retriever.py
│   └── test_chain.py
├── docs/
│   ├── arquitetura.drawio
│   └── relatorio.pdf
├── build_index.py            # script de ingestão: raw/ → limpeza → chunks → FAISS
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 12. Critérios de Sucesso

| Critério                        | Meta                          |
|---------------------------------|-------------------------------|
| Cobertura de intenções          | ≥ 6 fluxos funcionando        |
| Faithfulness (RAGAS)            | ≥ 0.85                        |
| Answer Relevancy (RAGAS)        | ≥ 0.80                        |
| Context Precision (RAGAS)       | ≥ 0.75                        |
| Context Recall (RAGAS)          | ≥ 0.75                        |
| Tempo de resposta               | < 8 segundos por query        |
| Escalonamento correto           | Detecta "falar com humano"    |

---

## 13. Relatório PDF — Estrutura

1. **Introdução** — Problema, contexto, motivação, segmento property
2. **Solução Proposta** — Arquitetura property.ai, tecnologias, justificativas
3. **Base de Dados** — Fontes, curadoria, estrutura do KB
4. **RAG Pipeline** — Detalhamento técnico com diagrama
5. **Avaliação** — Golden set, métricas RAGAS, resultados obtidos
6. **Fluxos Conversacionais** — Intenções e exemplos de diálogo
7. **Resultados** — Screenshots, casos de teste, evidências
8. **Conclusão** — Lições aprendidas, próximos passos
9. **Anexos** — Código-fonte comentado, diagramas

---

## 14. Checklist de Entrega

- [ ] Código-fonte completo no repositório
- [ ] `requirements.txt` atualizado
- [ ] Diagrama de arquitetura (draw.io ou PNG)
- [ ] Base de dados / KB documentado
- [ ] Golden set e resultados RAGAS
- [ ] Evidências de execução (screenshots ou vídeo)
- [ ] Relatório em PDF
- [ ] E-mail enviado por Fabricio Oliveira com CC para Millene Gomes, Sueli da Hora Moreira, Raquel Alcantara e David Fagundes
- [ ] Assunto: `InsurMinds - Atividade obrigatória 2`
- [ ] Enviado em **26/05/2026**
