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

def convert_json_embedding():
    logger.info("Mengonversi properti 'embedding' dari JSON string ke list...")
    
    sample_embedding = graph.run("MATCH (n:Movie) RETURN n.embedding LIMIT 1").evaluate()
    
    if isinstance(sample_embedding, str):
        graph.run("""
            MATCH (n:Movie)
            WHERE n.embedding IS NOT NULL AND apoc.meta.type(n.embedding) = 'String'
            SET n.embedding = apoc.convert.fromJsonList(n.embedding)
        """)
        logger.info("Konversi embedding selesai.")
    else:
        logger.info("Embeddings sudah dalam format yang benar, melewati konversi.")

def create_graph_projection():
    graph_name = "movieGraph"
    logger.info(f"Memeriksa dan membuat ulang graph projection '{graph_name}'...")
    if graph.run(f"CALL gds.graph.exists('{graph_name}') YIELD exists").evaluate("exists"):
        graph.run(f"CALL gds.graph.drop('{graph_name}')")
        logger.info(f"Graph projection '{graph_name}' dihapus.")

    # Memproyeksikan semua film
    graph.run(f"""
        CALL gds.graph.project(
            '{graph_name}',
            'Movie',
            '*',
            {{
                nodeProperties: ['embedding']
            }}
        )
    """)
    logger.info(f"Graph projection '{graph_name}' dibuat untuk semua film dengan embedding sinopsis.")

def run_knn(top_k=5):
    logger.info(f"Menjalankan algoritma KNN untuk mencari {top_k} film terdekat untuk setiap film...")
    
    # Menjalankan KNN menggunakan embedding sinopsis
    graph.run(f"""
        CALL gds.knn.write(
            'movieGraph',
            {{
                nodeProperties: ['embedding'],
                topK: {top_k},
                sampleRate: 1.0,
                deltaThreshold: 0.001,
                maxIterations: 10,
                writeRelationshipType: 'SIMILAR_TO',
                writeProperty: 'similarity_score',
                concurrency: 4
            }}
        )
    """)
    logger.info("KNN dengan embedding sinopsis selesai dijalankan.")

def calculate_weighted_similarity(synopsis_weight=0.40, actor_weight=0.30, genre_weight=0.30):
    """
    Menghitung similarity dengan pembagian bobot yang dapat disesuaikan:
    - embedding untuk sinopsis (text similarity)
    - graph_embedding untuk struktur relasi (aktor dan genre)
    
    Args:
        synopsis_weight: Bobot untuk kesamaan sinopsis (embedding)
        actor_weight: Bobot untuk kesamaan aktor
        genre_weight: Bobot untuk kesamaan genre
    """
    logger.info(f"Menghitung similarity dengan bobot: Sinopsis={synopsis_weight}, Aktor={actor_weight}, Genre={genre_weight}")

    # Query untuk mendapatkan data dari Neo4j
    query = """
    MATCH (n1:Movie)
    MATCH (n2:Movie)
    WHERE n1 <> n2  // Memastikan film tidak direkomendasikan ke dirinya sendiri
    OPTIONAL MATCH (n1)-[r1:SIMILAR_TO]->(n2)  // Embedding untuk sinopsis
    WHERE r1 IS NOT NULL  // Memastikan ada relasi similarity
    WITH n1, n2, r1,
         [(n1)-[:HAS_GENRE]->(g1) | g1.name] AS genres1,
         [(n2)-[:HAS_GENRE]->(g2) | g2.name] AS genres2,
         [(n1)-[:FEATURES]->(a1) | a1.name] AS actors1,
         [(n2)-[:FEATURES]->(a2) | a2.name] AS actors2
    RETURN 
        n1.id AS id1,
        n1.title AS title1,
        n1.overview AS overview1,
        n1.type AS type1,
        n2.id AS id2,
        n2.title AS title2,
        n2.overview AS overview2,
        n2.type AS type2,
        r1.similarity_score AS synopsis_sim,  // Skor dari embedding untuk sinopsis
        apoc.coll.intersection(genres1, genres2) AS common_genres,
        size(apoc.coll.intersection(genres1, genres2)) AS genre_overlap,
        apoc.coll.intersection(actors1, actors2) AS common_actors,
        size(apoc.coll.intersection(actors1, actors2)) AS actor_overlap,
        size(genres1) AS genres1_count,
        size(genres2) AS genres2_count,
        size(actors1) AS actors1_count,
        size(actors2) AS actors2_count
    """

    results = graph.run(query).data()

    if not results:
        logger.warning("Tidak ada hasil similarity yang ditemukan.")
        return []

    # Temukan nilai maksimum overlap genre dan aktor untuk normalisasi
    max_genre_overlap = max((row["genre_overlap"] for row in results if row["genre_overlap"] is not None), default=1)
    max_actor_overlap = max((row["actor_overlap"] for row in results if row["actor_overlap"] is not None), default=1)
    
    # Hindari pembagian dengan nol
    max_genre_overlap = max(max_genre_overlap, 1)
    max_actor_overlap = max(max_actor_overlap, 1)

    weighted_results = []

    for row in results:
        # Skor sinopsis dari embedding
        synopsis_sim = row['synopsis_sim'] or 0.0
        
        # Hitung skor genre menggunakan Jaccard similarity untuk menormalkan berdasarkan ukuran set
        genre_overlap = row['genre_overlap'] or 0
        genres1_count = row['genres1_count'] or 0
        genres2_count = row['genres2_count'] or 0
        genre_union_size = genres1_count + genres2_count - genre_overlap
        genre_sim = genre_overlap / max(genre_union_size, 1)  # Jaccard similarity
        
        # Hitung skor aktor menggunakan Jaccard similarity
        actor_overlap = row['actor_overlap'] or 0
        actors1_count = row['actors1_count'] or 0
        actors2_count = row['actors2_count'] or 0
        actor_union_size = actors1_count + actors2_count - actor_overlap
        actor_sim = actor_overlap / max(actor_union_size, 1)  # Jaccard similarity

        # Gabungkan semua skor dengan bobot yang ditentukan
        # Pastikan total bobot = 1.0
        total_weight = synopsis_weight + actor_weight + genre_weight
        if abs(total_weight - 1.0) > 0.01:  # Toleransi untuk kesalahan floating point
            # Normalisasi bobot jika totalnya tidak 1.0
            synopsis_weight /= total_weight
            actor_weight /= total_weight
            genre_weight /= total_weight
        
        # Hitung skor akhir dengan bobot yang ditentukan
        similarity = (
            synopsis_weight * synopsis_sim + 
            actor_weight * actor_sim + 
            genre_weight * genre_sim
        )

        if similarity > 0:
            weighted_results.append({
                "id1": row["id1"],
                "title1": row["title1"],
                "type1": row["type1"],
                "id2": row["id2"],
                "title2": row["title2"],
                "type2": row["type2"],
                "overview2": row["overview2"],
                "synopsis_sim": synopsis_sim,
                "actor_sim": actor_sim,
                "genre_sim": genre_sim,
                "similarity": similarity
            })

    return weighted_results

def create_similarity_relationships(results, top_n=5):
    """Membuat relationship WEIGHTED_SIMILAR_TO dengan properti yang mencakup semua skor similarity"""
    logger.info(f"Membuat relationship WEIGHTED_SIMILAR_TO dengan top {top_n} film terdekat...")
    
    # Hapus relasi WEIGHTED_SIMILAR_TO yang sudah ada di database
    graph.run("MATCH ()-[r:WEIGHTED_SIMILAR_TO]->() DELETE r")
    
    # Membuat dataframe untuk memudahkan pengelompokan dan pengurutan
    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("Tidak ada hasil untuk simpan sebagai relationships.")
        return
    
    df['similarity'] = df['similarity'].astype(float)
    
    # Kelompokkan berdasarkan id1 dan ambil top_n untuk setiap film
    relationships = []
    for id1, group in df.groupby('id1'):
        top_films = group.sort_values(by='similarity', ascending=False).head(top_n)
        for _, row in top_films.iterrows():
            relationships.append({
                'id1': row['id1'],
                'id2': row['id2'],
                'total_score': float(row['similarity']),
                'synopsis_score': float(row['synopsis_sim']),
                'actor_score': float(row['actor_sim']),
                'genre_score': float(row['genre_sim'])
            })
    
    # Batch insert relationships
    batch_size = 1000
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i+batch_size]
        graph.run("""
            UNWIND $relationships AS rel
            MATCH (n1:Movie {id: rel.id1})
            MATCH (n2:Movie {id: rel.id2})
            MERGE (n1)-[r:WEIGHTED_SIMILAR_TO]->(n2)
            SET r.weighted_score = rel.total_score,
                r.synopsis_score = rel.synopsis_score,
                r.actor_score = rel.actor_score,
                r.genre_score = rel.genre_score
        """, relationships=batch)
    
    logger.info(f"Berhasil membuat {len(relationships)} relationship WEIGHTED_SIMILAR_TO.")

def save_recommendations(results, top_n=5):
    logger.info(f"Menyimpan top {top_n} rekomendasi untuk setiap film ke dalam file JSON...")

    df = pd.DataFrame(results)
    if df.empty:
        logger.warning("Tidak ada hasil untuk disimpan.")
        return

    df['similarity'] = df['similarity'].astype(float)

    # Group by id1 dan sort by similarity
    recommendations = {}
    for id1, group in df.groupby('id1'):
        top_n_films = group.sort_values(by='similarity', ascending=False).head(top_n)
        recommendations[str(id1)] = [{
            "id": row['id2'],
            "title": row['title2'],
            "type": row['type2'],
            "overview": row['overview2'],
            "total_similarity_score": float(row['similarity']),
            "synopsis_score": float(row['synopsis_sim']),
            "actor_score": float(row['actor_sim']),
            "genre_score": float(row['genre_sim'])
        } for _, row in top_n_films.iterrows()]

    # Tentukan path hasil relatif terhadap script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "Result")
    os.makedirs(result_dir, exist_ok=True)

    sim_path = os.path.join(result_dir, "similarity_all_movies.json")
    flat_path = os.path.join(result_dir, "Rekomendasi.json")

    with open(sim_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4, ensure_ascii=False)
    logger.info(f"File {sim_path} berhasil disimpan.")

    # Format flat
    flat_recommendations = []
    for film_id, recs in recommendations.items():
        for rec in recs:
            flat_recommendations.append({
                "source_id": int(film_id),
                "id": rec["id"],
                "title": rec["title"],
                "overview": rec["overview"],
                "total_similarity_score": rec["total_similarity_score"],
                "synopsis_score": rec["synopsis_score"],
                "actor_score": rec["actor_score"],
                "genre_score": rec["genre_score"]
            })

    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(flat_recommendations, f, indent=4, ensure_ascii=False)
    logger.info(f"File {flat_path} berhasil disimpan (format flat).")

def cleanup_graph():
    try:
        graph.run("CALL gds.graph.drop('movieGraph')")
        logger.info("Graph projection dibersihkan.")
    except Exception as e:
        logger.error(f"Error saat membersihkan graph projection: {e}")

def main(top_k=5, synopsis_weight=0.40, actor_weight=0.30, genre_weight=0.30):
    try:
        convert_json_embedding()
        create_graph_projection()
        run_knn(top_k)
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
    # Anda dapat mengubah parameter ini sesuai kebutuhan
    main(
        top_k=5,              # Jumlah rekomendasi per film
        synopsis_weight=0.40, # Bobot untuk kesamaan sinopsis (menggunakan embedding)
        actor_weight=0.30,    # Bobot untuk kesamaan aktor
        genre_weight=0.30     # Bobot untuk kesamaan genre
    )