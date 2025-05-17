import logging
import pandas as pd
import json
from py2neo import Graph
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

def convert_all_embeddings():
    """Konversi semua embedding dari string JSON ke list"""
    logger.info("Memeriksa dan mengonversi format embedding jika diperlukan...")
    for prop in ['embedding', 'actor_embedding', 'genre_embedding']:
        sample = graph.run(f"MATCH (n:Movie) WHERE n.{prop} IS NOT NULL RETURN n.{prop} LIMIT 1").evaluate()
        if isinstance(sample, str):
            logger.info(f"Mengonversi {prop} dari string ke list...")
            graph.run(f"""
                MATCH (n:Movie)
                WHERE n.{prop} IS NOT NULL AND apoc.meta.type(n.{prop}) = 'String'
                SET n.{prop} = apoc.convert.fromJsonList(n.{prop})
            """)

def create_graph_projection():
    """Membuat ulang graph projection di GDS"""
    graph_name = "movieGraph"
    logger.info(f"Menyiapkan graph projection '{graph_name}'...")
    if graph.run(f"CALL gds.graph.exists('{graph_name}') YIELD exists").evaluate("exists"):
        graph.run(f"CALL gds.graph.drop('{graph_name}')")
        logger.info(f"Graph '{graph_name}' dihapus.")

    graph.run(f"""
        CALL gds.graph.project(
            '{graph_name}',
            ['Movie', 'Actor', 'Genre'],
            {{
                FEATURES: {{}},
                HAS_GENRE: {{}}
            }},
            {{
                nodeProperties: ['embedding', 'actor_embedding', 'genre_embedding']
            }}
        )
    """)
    logger.info(f"Graph projection '{graph_name}' berhasil dibuat.")

def calculate_weighted_similarity(synopsis_weight=0.1, actor_weight=0.1, genre_weight=0.8):
    """Menghitung similarity gabungan dengan skor yang di-threshold"""
    logger.info("Menghitung similarity gabungan dengan threshold...")

    query = """
    MATCH (n1:Movie), (n2:Movie)
    WHERE n1 <> n2
    WITH n1, n2,
         gds.similarity.cosine(n1.embedding, n2.embedding) AS synopsis_sim,
         gds.similarity.cosine(n1.actor_embedding, n2.actor_embedding) AS actor_sim,
         gds.similarity.cosine(n1.genre_embedding, n2.genre_embedding) AS genre_sim
    RETURN 
        n1.id AS id1, n1.title AS title1, n1.type AS type1, n1.overview AS overview1,
        n2.id AS id2, n2.title AS title2, n2.type AS type2, n2.overview AS overview2,
        synopsis_sim, actor_sim, genre_sim
    """

    results = graph.run(query).data()
    weighted_results = []

    for row in results:
        synopsis_sim = max(0, row['synopsis_sim'])  # Nilai negatif dijadikan 0
        actor_sim = max(0, row['actor_sim'])
        genre_sim = max(0, row['genre_sim'])

        total_sim = (
            synopsis_weight * synopsis_sim +
            actor_weight * actor_sim +
            genre_weight * genre_sim
        )

        row['synopsis_sim'] = synopsis_sim
        row['actor_sim'] = actor_sim
        row['genre_sim'] = genre_sim
        row['similarity'] = total_sim

        weighted_results.append(row)

    return weighted_results

def create_similarity_relationships(results, top_n=5):
    """Membuat relationship WEIGHTED_SIMILAR_TO berdasarkan top similarity"""
    logger.info("Membuat relationship WEIGHTED_SIMILAR_TO...")

    graph.run("MATCH ()-[r:WEIGHTED_SIMILAR_TO]->() DELETE r")

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("Data similarity kosong, tidak ada relationship yang dibuat.")
        return

    df['similarity'] = df['similarity'].astype(float)

    relationships = []
    for id1, group in df.groupby('id1'):
        top_films = group.sort_values(by='similarity', ascending=False).head(top_n)
        for _, row in top_films.iterrows():
            relationships.append({
                'id1': row['id1'],
                'id2': row['id2'],
                'total_score': row['similarity'],
                'synopsis_score': row['synopsis_sim'],
                'actor_score': row['actor_sim'],
                'genre_score': row['genre_sim']
            })

    for i in range(0, len(relationships), 1000):
        batch = relationships[i:i+1000]
        graph.run("""
            UNWIND $relationships AS rel
            MATCH (a:Movie {id: rel.id1}), (b:Movie {id: rel.id2})
            MERGE (a)-[r:WEIGHTED_SIMILAR_TO]->(b)
            SET r.weighted_score = rel.total_score,
                r.synopsis_score = rel.synopsis_score,
                r.actor_score = rel.actor_score,
                r.genre_score = rel.genre_score
        """, relationships=batch)

    logger.info(f"Relationship WEIGHTED_SIMILAR_TO berhasil dibuat sebanyak {len(relationships)}.")

def save_recommendations(results, top_n=5):
    """Menyimpan hasil rekomendasi ke file JSON"""
    logger.info(f"Menyimpan rekomendasi ke file JSON...")

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("Data hasil kosong, tidak ada yang disimpan.")
        return

    recommendations = {}
    for id1, group in df.groupby("id1"):
        top_recs = group.sort_values(by="similarity", ascending=False).head(top_n)
        recommendations[str(id1)] = [dict(
            id=row['id2'],
            title=row['title2'],
            type=row.get('type2'),
            overview=row.get('overview2'),
            total_similarity_score=row['similarity'],
            synopsis_score=row['synopsis_sim'],
            actor_score=row['actor_sim'],
            genre_score=row['genre_sim']
        ) for _, row in top_recs.iterrows()]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "Result")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "similarity_all_movies.json"), "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)

    flat = []
    for source_id, recs in recommendations.items():
        for r in recs:
            flat.append({
                "source_id": int(source_id),
                **r
            })

    with open(os.path.join(result_dir, "Rekomendasi.json"), "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=4, ensure_ascii=False)

    logger.info("File rekomendasi berhasil disimpan.")

def cleanup_graph():
    """Membersihkan graph projection"""
    try:
        exists = graph.run("CALL gds.graph.exists('movieGraph') YIELD exists").evaluate()
        if exists:
            graph.run("CALL gds.graph.drop('movieGraph')")
            logger.info("Graph projection berhasil dihapus.")
    except Exception as e:
        logger.error(f"Gagal membersihkan graph projection: {e}")

def main(top_k=5, synopsis_weight=0.1, actor_weight=0.1, genre_weight=0.8):
    try:
        convert_all_embeddings()
        create_graph_projection()
        results = calculate_weighted_similarity(synopsis_weight, actor_weight, genre_weight)

        if results:
            create_similarity_relationships(results, top_k)
            save_recommendations(results, top_k)
        else:
            logger.warning("Tidak ada hasil similarity yang valid.")
    except Exception as e:
        logger.error(f"Terjadi error utama: {e}", exc_info=True)
    finally:
        cleanup_graph()

if __name__ == "__main__":
    main(
        top_k=5,
        synopsis_weight=0.2,
        actor_weight=0.7,
        genre_weight=0.1
    )
