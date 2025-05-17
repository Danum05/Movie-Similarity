import logging
import time
import os
from py2neo import Graph
import numpy as np
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Node2Vec")

# Koneksi ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Set path file hasil
base_dir = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_FILE = os.path.join(base_dir, 'Result', 'Node2Vec_embeddings.csv')

GRAPH_NAME = "movieGraph"
EMBEDDING_PROPERTY = "graph_embedding"

def drop_existing_graph(graph):
    logger.info(f"Memeriksa graf '{GRAPH_NAME}'...")
    result = graph.run(f"CALL gds.graph.exists('{GRAPH_NAME}') YIELD exists RETURN exists").data()
    if result and result[0]['exists']:
        logger.info(f"Menghapus graf '{GRAPH_NAME}' yang sudah ada...")
        graph.run(f"CALL gds.graph.drop('{GRAPH_NAME}', false) YIELD graphName")
    logger.info("Graf siap dibuat ulang.")

def create_graph_projection(graph):
    logger.info("Membuat graph projection...")
    graph.run(f"""
        CALL gds.graph.project(
            '{GRAPH_NAME}',
            ['Movie', 'Genre', 'Actor'],
            {{
                HAS_GENRE: {{ orientation: 'UNDIRECTED' }},
                FEATURES: {{ orientation: 'UNDIRECTED' }}
            }}
        )
    """)
    logger.info("Graph projection berhasil dibuat.")

def run_node2vec(graph):
    logger.info("Menjalankan algoritma Node2Vec...")
    result = graph.run(f"""
        CALL gds.beta.node2vec.stream('{GRAPH_NAME}', {{
            embeddingDimension: 128,
            walkLength: 80,
            returnFactor: 1.0,
            inOutFactor: 1.0
        }})
        YIELD nodeId, embedding
        WITH nodeId, embedding
        MATCH (n) WHERE id(n) = nodeId AND (n:Movie OR n:Genre OR n:Actor)
        RETURN n.id AS id, embedding, labels(n)[0] AS label
    """).data()
    logger.info(f"Node2Vec selesai dijalankan. Total node: {len(result)}")
    return result

def normalize_and_store_embeddings(graph, records):
    valid_embeddings = []

    for record in records:
        node_id = record['id']
        embedding = np.array(record['embedding'])

        if embedding is None or np.all(embedding == 0):
            logger.warning(f"Skipping node ID {node_id}: embedding kosong atau nol.")
            continue

        try:
            normalized = embedding / np.linalg.norm(embedding)
            if record['label'] == 'Movie':
                graph.run("""
                    MATCH (n:Movie {id: $id})
                    SET n.graph_embedding = $embedding
                """, id=node_id, embedding=normalized.tolist())

                valid_embeddings.append({
                    'id': node_id,
                    'embedding': normalized.tolist(),
                    'type': 'Movie'
                })

            logger.info(f"Embedding sukses untuk {record['label']} ID: {node_id}")

        except Exception as e:
            logger.error(f"Error saat memproses node ID {node_id}: {e}")

    return valid_embeddings

def save_embeddings(embeddings):
    if not embeddings:
        logger.warning("Tidak ada embedding valid untuk disimpan.")
        return

    df = pd.DataFrame(embeddings)
    df.to_csv(EMBEDDING_FILE, index=False)
    logger.info(f"Embeddings disimpan ke '{EMBEDDING_FILE}'")

    embedding_values = np.array([e['embedding'] for e in embeddings])
    logger.info("Statistik embedding:")
    logger.info(f"Shape: {embedding_values.shape}")
    logger.info(f"Mean: {np.mean(embedding_values):.4f}")
    logger.info(f"Std: {np.std(embedding_values):.4f}")
    logger.info(f"Min: {np.min(embedding_values):.4f}")
    logger.info(f"Max: {np.max(embedding_values):.4f}")

def main():
    total_start = time.time()

    drop_existing_graph(graph)
    create_graph_projection(graph)
    records = run_node2vec(graph)
    embeddings = normalize_and_store_embeddings(graph, records)
    save_embeddings(embeddings)

    logger.info(f"Total waktu eksekusi Node2Vec: {time.time() - total_start:.2f} detik")

if __name__ == "__main__":
    main()
