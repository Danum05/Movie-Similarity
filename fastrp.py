import logging
import time
import os
from py2neo import Graph
import numpy as np
import pandas as pd

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FastRP")

# Koneksi ke database Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# Konfigurasi path dan nama graph
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTOR_EMBEDDING_FILE = os.path.join(BASE_DIR, 'Result', 'actor_embeddings.csv')
GENRE_EMBEDDING_FILE = os.path.join(BASE_DIR, 'Result', 'genre_embeddings.csv')

GRAPH_NAMES = {
    'actor': 'actorGraph',
    'genre': 'genreGraph'
}

def drop_existing_graphs():
    """Hapus graph projection yang sudah ada"""
    for gtype in GRAPH_NAMES.values():
        logger.info(f"Memeriksa graph '{gtype}'...")
        result = graph.run(f"CALL gds.graph.exists('{gtype}') YIELD exists RETURN exists").data()
        if result and result[0]['exists']:
            logger.info(f"Menghapus graph '{gtype}'...")
            graph.run(f"CALL gds.graph.drop('{gtype}', false)")

def create_graph_projection(relationship_type):
    """Buat graph projection terpisah untuk aktor dan genre"""
    graph_name = GRAPH_NAMES[relationship_type.lower()]
    target_relationship = 'FEATURES' if relationship_type == 'actor' else 'HAS_GENRE'
    
    logger.info(f"Membuat {relationship_type} graph projection...")
    graph.run(f"""
        CALL gds.graph.project(
            '{graph_name}',
            ['Movie', '{relationship_type.capitalize()}'],
            {{
                {target_relationship}: {{
                    orientation: 'UNDIRECTED'
                }}
            }}
        )
    """)

def run_fastrp(relationship_type):
    """Jalankan FastRP untuk masing-masing graph"""
    graph_name = GRAPH_NAMES[relationship_type]
    logger.info(f"Menjalankan FastRP untuk {relationship_type}...")
    
    result = graph.run(f"""
        CALL gds.fastRP.stream('{graph_name}', {{
            embeddingDimension: 128,
            iterationWeights: [0.0, 1.0, 10.0],
            propertyRatio: 0.0,
            normalizationStrength: 0.8,
            randomSeed: 1
        }})
        YIELD nodeId, embedding
        WHERE gds.util.asNode(nodeId):Movie
        RETURN gds.util.asNode(nodeId).id AS movieId, embedding
    """).data()
    
    logger.info(f"Berhasil memproses {len(result)} film untuk {relationship_type}")
    return result

def process_embeddings(embeddings, relationship_type):
    """Normalisasi dan simpan embeddings"""
    valid_embeddings = []
    prop_name = f"{relationship_type}_embedding"
    
    for record in embeddings:
        try:
            movie_id = record['movieId']
            embedding = np.array(record['embedding'])
            normalized = embedding / np.linalg.norm(embedding)
            
            # Simpan ke database
            graph.run(f"""
                MATCH (m:Movie {{id: $id}})
                SET m.{prop_name} = $embedding
            """, id=movie_id, embedding=normalized.tolist())
            
            valid_embeddings.append({
                'movieId': movie_id,
                'embedding': normalized.tolist(),
                'type': relationship_type
            })
            
        except Exception as e:
            logger.error(f"Error processing {movie_id}: {str(e)}")
    
    return valid_embeddings

def save_to_csv(embeddings, relationship_type):
    """Simpan embeddings ke CSV"""
    df = pd.DataFrame(embeddings)
    file_path = ACTOR_EMBEDDING_FILE if relationship_type == 'actor' else GENRE_EMBEDDING_FILE
    df.to_csv(file_path, index=False)
    logger.info(f"Embeddings {relationship_type} disimpan di {file_path}")

def main():
    total_start = time.time()
    
    try:
        # 1. Hapus graph yang ada
        drop_existing_graphs()
        
        # 2. Proses untuk aktor dan genre secara terpisah
        for relationship_type in ['actor', 'genre']:
            # Buat graph projection
            create_graph_projection(relationship_type)
            
            # Jalankan FastRP
            raw_embeddings = run_fastrp(relationship_type)
            
            # Proses dan simpan embeddings
            processed = process_embeddings(raw_embeddings, relationship_type)
            save_to_csv(processed, relationship_type)
            
    except Exception as e:
        logger.error(f"Error utama: {str(e)}")
    finally:
        logger.info(f"Total waktu eksekusi: {time.time() - total_start:.2f} detik")

if __name__ == "__main__":
    main()