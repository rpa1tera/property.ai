# INSTALL — property.ai

Testado em: Windows 11, Python 3.11, PowerShell.

---

## Pré-requisitos

**1. Python 3.11+** — https://www.python.org/downloads/

**2. Visual C++ Redistributable 2022 x64** — obrigatório para ChromaDB:  
https://aka.ms/vs/17/release/vc_redist.x64.exe  
Instale e reinicie o terminal antes de continuar.

**3. Chave Groq (gratuita)** — https://console.groq.com

---

## Instalação

```powershell
cd G:\PYTHON\property.ai_raquel
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Se o PowerShell bloquear scripts: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Verificar ambiente

```powershell
python teste_chroma.py
```

Resultado esperado:
```
1. carregando embeddings (download ~280MB na 1a vez)...
   embeddings ok — dim=768
2. testando chromadb com HuggingFace EF...
   chroma ok — count=1
TUDO OK — pode continuar.
```

---

## Configurar variáveis de ambiente

```powershell
copy .env.example .env
```

Edite `.env`:
```
GROQ_API_KEY=gsk_SUACHAVEAQUI
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=intfloat/multilingual-e5-base
CHROMA_PERSIST_DIR=./data/processed/embeddings
RETRIEVER_TOP_K=5
```

---

## Construir o índice

Processa os documentos de `data/raw/` e indexa no ChromaDB. Execute uma vez (ou novamente ao adicionar documentos).

```powershell
python build_index.py
```

---

## Rodar testes

```powershell
python -m pytest tests/ -v
```

Esperado: **13 passed**.

---

## Iniciar o chatbot

```powershell
streamlit run src/ui/app.py
```

Abre em `http://localhost:8501`.

---

## Comandos rápidos

```powershell
.venv\Scripts\Activate.ps1          # ativar venv
python build_index.py               # reconstruir índice
python -m pytest tests/ -v          # rodar testes
streamlit run src/ui/app.py         # iniciar chatbot
```

---

## Solução de problemas

**ChromaDB trava / `STATUS_ACCESS_VIOLATION`**  
Reinstale o Visual C++ Redistributable e reinicie o terminal.

**`GROQ_API_KEY` não encontrada**  
Verifique se `.env` existe na raiz e contém `GROQ_API_KEY=gsk_...`.

**PowerShell não executa `Activate.ps1`**  
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
