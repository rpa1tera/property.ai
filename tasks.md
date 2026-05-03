← [README](README.md) · [SPEC](SPEC.md)

# Tasks — property.ai (InsurMinds ChatBot)

**Prazo final:** 26/05/2026  
**Chatbot congelado:** 22/05/2026 (Fabricio Oliveira assume a documentação a partir daqui)

## Equipe

| Pessoa | Perfil | Área |
|---|---|---|
| Millene Gomes | TI | Pipeline de Dados (ingestão, chunking, ChromaDB) |
| Sueli da Hora Moreira | TI | Interface Streamlit (Full Stack) |
| Raquel Alcantara | TI | LLM / LangChain / Groq |
| David Fagundes | Negócios | Regras de negócio + bases de dados + golden set |
| Fabricio Oliveira | Negócios | Documentação, relatório, entrega |

---

## Fase 1 — Fundação (01/05 – 08/05)

| # | Tarefa | Responsável | Depende de | Status |
|---|---|---|---|---|
| 1.1 | Criar repositório, estrutura de pastas e `.env.example` | Raquel Alcantara | — | [x] |
| 1.2 | Levantar e organizar FAQs do segmento property (mínimo 20 Q&A em PT-BR) | David Fagundes | — | [ ] |
| 1.3 | Providenciar 5–10 PDFs de apólices/manuais property (residencial, empresarial ou condomínio) | David Fagundes | — | [ ] |
| 1.4 | Implementar `corpus_loader.py` — detecta formato pelo sufixo (.pdf → PyMuPDF, .html → BSHTMLLoader, .csv → pandas, .json → JSONLoader) e retorna lista de `Document` | Millene Gomes | 1.2 + 1.3 | [ ] |
| 1.5 | Implementar `cleaner.py` — limpeza de cabeçalhos/rodapés, hifenização, encoding PT-BR, duplicatas, respostas curtas | Millene Gomes | 1.4 | [ ] |
| 1.6 | Implementar `chunker.py` — quebra por sentença `[.!?]` primeiro, depois agrupa por tamanho (512 tokens, overlap 64) + metadados | Millene Gomes | 1.5 | [ ] |
| 1.7 | Indexar tudo no ChromaDB com `multilingual-e5-base` (`embeddings.py`) | Millene Gomes | 1.6 | [ ] |
| 1.8 | Validar recuperação semântica com 10 queries de teste em PT-BR | Raquel Alcantara + David Fagundes | 1.7 | [ ] |

### Notas de implementação — Fase 1

**1.4 `corpus_loader.py` (Millene Gomes):**

- PDF: `import fitz` (PyMuPDF) — `fitz.open(path)`, iterar páginas com `page.get_text()`
- HTML: `from langchain_community.document_loaders import BSHTMLLoader` — requer `pip install beautifulsoup4`
- CSV: `import pandas as pd` — `pd.read_csv(path)`, validar colunas `pergunta` e `resposta`
- JSON: `from langchain_community.document_loaders import JSONLoader`
- Retornar sempre uma lista de objetos `Document` do LangChain: `from langchain_core.documents import Document`

**1.7 `embeddings.py` (Millene Gomes):**

- Usar `HuggingFaceEmbeddings` e `Chroma` do LangChain:

  ```python
  from langchain_huggingface import HuggingFaceEmbeddings
  from langchain_chroma import Chroma
  embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
  vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="data/processed/embeddings")
  ```

- **Atenção:** o modelo baixa ~280MB na primeira execução — é normal, não é erro.

---

## Fase 2 — Core ChatBot (09/05 – 13/05)

| # | Tarefa | Responsável | Depende de | Status |
|---|---|---|---|---|
| 2.1 | Configurar `ChatGroq` e testar conexão com `llama-3.3-70b-versatile` | Raquel Alcantara | 1.1 | [ ] |
| 2.2 | Definir regras de resposta do property.ai (o que pode/não pode dizer, tom, limites) | David Fagundes | — | [ ] |
| 2.3 | Implementar `rag_fusion.py` — chama LLM para gerar 3 variações semânticas da query, busca cada uma no ChromaDB, deduplica e funde top-10 | Raquel Alcantara | 2.1 + 1.8 | [ ] |
| 2.4 | Construir `chain.py` — RAG chain com prompt estruturado `<chunks>/<query>` integrando RAG-Fusion + CRAG Evaluator | Raquel Alcantara | 2.3 | [ ] |
| 2.5 | Implementar `crag_evaluator.py` — LLM avalia se chunks são SUFICIENTES; se não, escala para humano em vez de alucinar | Raquel Alcantara | 2.4 | [ ] |
| 2.6 | Implementar `memory.py` — histórico conversacional | Raquel Alcantara | 2.4 | [ ] |
| 2.7 | Implementar `intents.py` — detecção de intenções + escalonamento | Raquel Alcantara | 2.2 + 2.4 | [ ] |
| 2.8 | Cobrir os 6 fluxos: cobertura, franquia, sinistro, apólice, cancelamento, escalonamento | Raquel Alcantara + David Fagundes | 2.7 | [ ] |
| 2.9 | Testes unitários do retriever, RAG-Fusion e CRAG Evaluator | Millene Gomes + Raquel Alcantara | 2.8 | [ ] |

---

## Fase 2.5 — Avaliação RAG (14/05 – 16/05, paralelo ao fim da Fase 2)

| # | Tarefa | Responsável | Depende de | Status |
|---|---|---|---|---|
| E.1 | Criar golden set: 20–30 pares pergunta / resposta esperada / documento fonte — foco em property | David Fagundes | 1.2 + 1.3 | [ ] |
| E.2 | Implementar `ragas_eval.py` — executa pipeline RAGAS sobre o golden set | Raquel Alcantara + Millene Gomes | E.1 + 2.3 | [ ] |
| E.3 | Rodar avaliação e coletar métricas: Faithfulness, Answer Relevancy, Context Precision, Context Recall | Raquel Alcantara | E.2 | [ ] |
| E.4 | Analisar resultados: se alguma métrica < meta, ajustar chunking (Millene Gomes) ou prompt (Raquel Alcantara) | Millene Gomes + Raquel Alcantara | E.3 | [ ] |
| E.5 | Documentar resultados RAGAS em tabela para o relatório | Fabricio Oliveira | E.3 | [ ] |

---

## Fase 3 — Interface e Refinamento (17/05 – 21/05)

| # | Tarefa | Responsável | Depende de | Status |
|---|---|---|---|---|
| 3.1 | Criar layout base da UI no Streamlit com identidade property.ai | Sueli da Hora Moreira | 2.3 | [ ] |
| 3.2 | Integrar chain RAG + memória conversacional à UI | Sueli da Hora Moreira + Raquel Alcantara | 3.1 + 2.4 | [ ] |
| 3.3 | Componente de exibição de fontes/documentos citados na resposta | Sueli da Hora Moreira | 3.2 | [ ] |
| 3.4 | Componente de escalonamento ("Falar com atendente") | Sueli da Hora Moreira | 3.2 + 2.5 | [ ] |
| 3.5 | Refinamento de prompts com base em testes manuais | Raquel Alcantara + David Fagundes | 3.2 | [ ] |
| 3.6 | Testes manuais de qualidade — David Fagundes valida se respostas fazem sentido no domínio property | David Fagundes | 3.2 | [ ] |
| 3.7 | Capturar evidências de execução (screenshots + gravação de tela) | Sueli da Hora Moreira + Fabricio Oliveira | 3.2 | [ ] |

### Notas de implementação — Fase 3

**3.1 Layout Streamlit (Sueli da Hora Moreira):**

- Usar `st.chat_message("user")` e `st.chat_message("assistant")` para bolhas de conversa
- Histórico deve ser armazenado em `st.session_state["messages"]` (lista de dicts com `role` e `content`)
- Inicializar no topo do script: `if "messages" not in st.session_state: st.session_state["messages"] = []`

**3.2 Integração da chain à UI (Sueli da Hora Moreira + Raquel Alcantara):**

- Chamar a chain com `chain.invoke({"question": user_input, "chat_history": st.session_state["messages"]})`
- Usar `st.chat_input("Digite sua pergunta...")` para capturar entrada do usuário

---

## Fase 4 — Documentação e Entrega (22/05 – 25/05) — Fabricio Oliveira lidera

| # | Tarefa | Responsável | Depende de | Status |
|---|---|---|---|---|
| 4.1 | Diagrama de arquitetura do property.ai (draw.io ou PNG) | Fabricio Oliveira | 3.x | [ ] |
| 4.2 | Redigir relatório | Fabricio Oliveira |
| 4.3 | Revisar e montar PDF final | Fabricio Oliveira | 4.2 | [ ] |
| 4.4 | Revisão geral do código e `requirements.txt` | Fabricio Oliveira | todas | [ ] |
| 4.5 | Revisão final da qualidade das respostas do chatbot | Fabricio Oliveira | 3.6 | [ ] |
| 4.6 | Enviar e-mail para <challenges@i2a2.academy> com todos os artefatos em **26/05** | Raquel Alcantara | 4.4 + 4.5 | [ ] |
