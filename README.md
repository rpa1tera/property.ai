# property.ai — ChatBot de Seguros Patrimoniais

ChatBot conversacional especializado no segmento **property** (seguros patrimoniais), desenvolvido com IA Generativa e RAG (Retrieval-Augmented Generation).

Projeto desenvolvido para o programa **InsurMinds — I2A2 Academy (Atividade Obrigatória 2)**.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![Groq](https://img.shields.io/badge/LLM-Groq_API-F55036)
![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-FF6B35)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Em_Desenvolvimento-yellow)

---

## Sobre o projeto

O **property.ai** automatiza o atendimento a segurados do ramo property, respondendo perguntas sobre coberturas, franquias, acionamento de sinistros e documentação com base em apólices e manuais reais.

Destaques técnicos:
- **RAG-Fusion** — gera variações semânticas da query para melhorar a recuperação
- **CRAG Evaluator** — avalia se o contexto recuperado é suficiente antes de responder
- **Embeddings multilíngues** — `intfloat/multilingual-e5-base`, otimizado para PT-BR
- **ChromaDB** — vector store com persistência automática e filtros por metadado
- **Groq API** — LLM gratuito (`llama-3.3-70b-versatile`) com excelente desempenho em português

---

## Pré-requisitos

- Python 3.11+
- Chave de API gratuita do Groq: [console.groq.com](https://console.groq.com)

---

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repo>
cd property.ai

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env e preencha GROQ_API_KEY
```

---

## Como rodar

```bash
# Indexar os documentos (primeira vez ou ao adicionar novos docs)
# Coloque os arquivos em data/raw/ antes de rodar
# Obs: na primeira execução o modelo de embeddings (~280MB) será baixado automaticamente
python src/ingestion/corpus_loader.py

# Iniciar o chatbot
streamlit run src/ui/app.py

# Rodar a avaliação RAGAS
python src/evaluation/ragas_eval.py

# Rodar os testes
pytest tests/
```

---

## Estrutura do projeto

```
property.ai/
├── data/
│   ├── raw/              # Documentos originais (FAQs, apólices, manuais)
│   ├── processed/        # Chunks e embeddings gerados — ignorado pelo git
│   └── evaluation/       # Golden set para avaliação RAGAS
├── src/
│   ├── ingestion/        # Extração, limpeza e chunking de documentos
│   ├── rag/              # Embeddings e retriever (ChromaDB)
│   ├── chatbot/          # Chain RAG, RAG-Fusion, CRAG Evaluator, memória
│   ├── evaluation/       # Pipeline de avaliação com RAGAS
│   └── ui/               # Interface Streamlit
├── tests/
├── docs/                 # Diagrama de arquitetura e relatório PDF
├── notebooks/            # EDA exploratório
├── .env.example
├── .gitignore
└── requirements.txt
```

---

## Equipe

| Nome | Área |
|---|---|
| Millene Gomes | Pipeline de Dados |
| Sueli da Hora Moreira | Interface Streamlit |
| Raquel Alcantara | LLM / LangChain / RAG |
| David Fagundes | Regras de Negócio + Base de Dados |
| Fabricio Oliveira | Documentação e Entrega |

---

## Documentação do projeto

| Documento | O que contém | Quando abrir |
|---|---|---|
| [`SPEC.md`](SPEC.md) | Arquitetura, stack, pipeline RAG, estratégia de dados, prompts, avaliação RAGAS, critérios de sucesso | Dúvida sobre "como funciona", "por que X foi escolhido" ou "qual é o comportamento esperado" |
| [`tasks.md`](tasks.md) | Tarefas por fase, responsáveis, dependências e status | Verificar o que fazer agora, quem depende de quê, ou atualizar o status de uma tarefa |

> Toda mudança de decisão técnica ou de escopo deve ser refletida nos dois documentos acima.
