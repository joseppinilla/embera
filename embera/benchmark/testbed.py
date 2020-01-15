import time
import embera

__all__ = ["embed_and_report"]

def embed_and_report(method, S, T):
    report = {}
    embedding = {}

    start = time.time()
    embedding = method(S,T)
    end = time.time()

    if not embedding: return {"valid":False}, {}
    embedding_obj = embera.Embedding(S,T,embedding)

    report["embedding_method"] = method.__name__
    report["embedding_runtime"] = end - start
    report["max_chain"] = embedding_obj.max_chain
    report["total_qubits"] = embedding_obj.total_qubits
    report["valid"] = True

    return embedding, report
