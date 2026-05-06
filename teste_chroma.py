import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class HFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, hf_embeddings):
        self._hf = hf_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return self._hf.embed_documents(input)


print('1. carregando embeddings (download ~280MB na 1a vez)...')
emb = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base',
                             model_kwargs={'device': 'cpu'})
v = emb.embed_query('franquia de incendio')
print(f'   embeddings ok — dim={len(v)}')

print('2. testando chromadb com HuggingFace EF...')
c = chromadb.EphemeralClient()
ef = HFEmbeddingFunction(emb)
col = c.create_collection('smoke_test', embedding_function=ef)
col.add(documents=['seguro patrimonial property'], ids=['t1'],
        metadatas=[{'tipo': 'teste'}])
print(f'   chroma ok — count={col.count()}')
print('TUDO OK — pode continuar.')
