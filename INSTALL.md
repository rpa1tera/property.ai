# INSTALL — property.ai

Guia completo de instalação em ambiente virtual isolado.  
Testado em: **Windows 11, Python 3.11, PowerShell**.

---

## Pré-requisitos do sistema

### 1. Python 3.11+
Verifique:
```powershell
python --version   # deve retornar 3.11.x ou superior
```
Download se necessário: https://www.python.org/downloads/

### 2. Visual C++ Redistributable 2022 x64 — OBRIGATÓRIO para ChromaDB
O ChromaDB usa bibliotecas nativas (`hnswlib`, `onnxruntime`) que dependem do runtime MSVC.  
**Instale antes de qualquer coisa:**

```
https://aka.ms/vs/17/release/vc_redist.x64.exe
```

Baixe, execute o instalador e **reinicie o terminal** após a instalação.

### 3. Chave da API Groq (gratuita)
Crie sua conta e gere uma chave em: https://console.groq.com  
Guarde — será usada no `.env` mais adiante.

---

## Passo a passo

### Passo 1 — Navegar até o projeto
```powershell
cd G:\PYTHON\property.ai_raquel
```

### Passo 2 — Criar o ambiente virtual
```powershell
python -m venv .venv
```

### Passo 3 — Ativar o ambiente virtual
```powershell
.venv\Scripts\Activate.ps1
```

Você verá `(.venv)` no início do prompt. **Todos os passos seguintes devem ser executados com o venv ativo.**

> Se o PowerShell bloquear a execução de scripts:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Passo 4 — Atualizar pip
```powershell
python -m pip install --upgrade pip
```

### Passo 5 — Instalar dependências
```powershell
pip install -r requirements.txt
```

Isso instala (entre outros):
- `torch==2.5.1+cpu` — versão CPU, sem dependência de NVIDIA
- `chromadb>=1.0.9` + `langchain-chroma>=0.2.4`
- `langchain-groq` + `langchain-huggingface`
- `sentence-transformers` (modelo multilingual-e5-base, ~280 MB no primeiro uso)
- `streamlit`, `pymupdf`, `ragas`, `pytest`

---

## Passo 6 — Testar o ambiente (CRÍTICO — faça antes de continuar)

```powershell
python teste_chroma.py
```

**Resultado esperado:**
```
1. carregando embeddings (download ~280MB na 1a vez)...
   embeddings ok — dim=768
2. testando chromadb com HuggingFace EF...
   chroma ok — count=1
TUDO OK — pode continuar.
```

**Se falhar**, veja a seção [Solução de problemas](#solução-de-problemas) ao final.

---

## Passo 7 — Configurar variáveis de ambiente

Copie o arquivo de exemplo:
```powershell
copy .env.example .env
```

Edite `.env` e preencha sua chave:
```
GROQ_API_KEY=gsk_SUACHAVEAQUI

GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_PERSIST_DIR=./data/processed/embeddings
RETRIEVER_TOP_K=5
```

---

## Passo 8 — Construir o índice vetorial

Este passo processa os PDFs de `data/raw/`, gera chunks e indexa no ChromaDB.  
**Execute apenas uma vez** (ou novamente se adicionar novos documentos).

```powershell
python build_index.py
```

Saída esperada:
```
[build_index] 2 arquivo(s) encontrado(s):
  Carregando: CG_RN_96_V3.2 - MAPFRE SEGUROS.pdf (tipo=apolice)
    -> 96 página(s)/entrada(s) extraída(s)
  Carregando: Introduçao ao Resseguro  Fundación MAPFRE.pdf (tipo=manual)
    -> 216 página(s)/entrada(s) extraída(s)
[build_index] Total bruto: 312 documentos
[build_index] Após limpeza: 312 documentos
[build_index] Chunks gerados: 2102
[build_index] Criando embeddings e indexando no ChromaDB...
  indexado 100/2102
  indexado 200/2102
  ...
[build_index] Indexação concluída: 2102 chunks
```

---

## Passo 9 — Executar os testes

```powershell
python -m pytest tests/ -v
```

Deve exibir **13 passed**.

---

## Passo 10 — Iniciar o chatbot

```powershell
streamlit run src/ui/app.py
```

O browser abre automaticamente em `http://localhost:8501`.  
Perguntas de exemplo para testar:

| Intenção | Pergunta |
|---|---|
| Cobertura | `Meu seguro cobre danos por enchente?` |
| Franquia | `Qual é a franquia para sinistro de incêndio?` |
| Sinistro | `Como aciono o seguro após um roubo na empresa?` |
| Apólice | `Como obter a 2ª via da minha apólice?` |
| Cancelamento | `Quero cancelar meu seguro, como faço?` |
| Escalonamento | `Quero falar com um atendente` |

---

## Estrutura de arquivos criada

Após a instalação, o projeto terá:
```
property.ai/
├── .venv/                        # ambiente virtual (não versionado)
├── data/
│   ├── raw/                      # PDFs originais
│   └── processed/
│       └── embeddings/           # índice ChromaDB gerado (não versionado)
├── src/
│   ├── ingestion/                # corpus_loader, cleaner, chunker
│   ├── rag/                      # embeddings (ChromaDB), retriever
│   ├── chatbot/                  # chain, rag_fusion, crag_evaluator, intents, memory
│   ├── evaluation/               # ragas_eval
│   └── ui/                       # app.py (Streamlit)
├── tests/                        # 13 testes pytest
├── build_index.py                # script de ingestão
├── .env                          # suas chaves (não versionado)
└── requirements.txt
```

---

## Comandos rápidos de referência

```powershell
# Ativar venv (sempre que abrir novo terminal)
.venv\Scripts\Activate.ps1

# Reconstruir índice (após adicionar novos documentos)
python build_index.py

# Rodar testes
python -m pytest tests/ -v

# Iniciar chatbot
streamlit run src/ui/app.py

# Avaliação RAGAS (requer golden_set.json completo)
python -c "from src.evaluation.ragas_eval import run_evaluation; print(run_evaluation())"
```

---

## Solução de problemas

### ChromaDB trava ou exibe `STATUS_ACCESS_VIOLATION`

**Causa:** DLL nativa do HNSW ou onnxruntime incompatível com o sistema.

**Solução 1 — Reinstalar o Visual C++ Redistributable**  
Baixe e instale novamente: https://aka.ms/vs/17/release/vc_redist.x64.exe  
Reinicie o terminal e repita o Passo 6.

**Solução 2 — Docker (100% confiável)**

Instale o Docker Desktop: https://www.docker.com/products/docker-desktop

Suba o servidor ChromaDB:
```powershell
docker run -d -p 8000:8000 `
  -v G:/PYTHON/chroma_data:/chroma/chroma `
  --name chromadb chromadb/chroma:latest
```

Avise a Raquel — ela atualiza `embeddings.py` para usar `HttpClient` em vez de `PersistentClient` (alteração de 2 linhas).

**Solução 3 — FAISS (fallback local, sem dependências nativas)**  
Avise a Raquel — o código com FAISS já está implementado e testado, basta reverter.

---

### `onnxruntime` falha ao importar

```
ImportError: DLL load failed while importing onnxruntime_pybind11_state
```

O ChromaDB usa onnxruntime para sua função de embedding padrão. No nosso código, esse problema já está contornado: passamos sempre nossa própria função HuggingFace ao ChromaDB, então o onnxruntime **nunca é chamado**. Se o erro persistir mesmo assim, instale o VC++ Redistributable (Solução 1 acima).

### `GROQ_API_KEY` não encontrada

```
GroqError: The api_key client option must be set...
```

Verifique se o `.env` existe na raiz do projeto e contém `GROQ_API_KEY=gsk_...`.

### Modelo de embeddings não baixa

Na primeira execução, o modelo `intfloat/multilingual-e5-base` (~280 MB) é baixado automaticamente do HuggingFace. Se a rede bloquear:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
```

### PowerShell não executa scripts (Activate.ps1)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
