import re
import json
import logging
import pandas as pd
from bs4 import BeautifulSoup
from py2neo import Graph
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import nltk
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "211524037"))

# === Text Preprocessing ===

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def case_folding(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"\d+", "", text)
    return text

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word not in stop_words)

def preprocess_text(text):
    text = remove_html_tags(text)
    text = case_folding(text)
    text = remove_stopwords(text)
    return text

# === Utility Functions ===

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_dataframe(data):
    clean_data = []
    for i, item in enumerate(data):
        try:
            if all(k in item for k in ("id", "title", "genre", "overview", "actors")):
                clean_data.append(item)
            else:
                logger.warning(f"Skipping invalid entry at index {i}: missing required fields")
        except Exception as e:
            logger.error(f"Error processing item at index {i}: {e}")

    if not clean_data:
        raise ValueError("No valid entries found in dataset.")

    return pd.DataFrame.from_dict(clean_data)[["id", "title", "genre", "overview", "actors"]]

def generate_embeddings(texts):
    return model.encode(texts, show_progress_bar=True).tolist()

# === Neo4j Insertion Functions ===

def create_constraints():
    graph.run("CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE")
    graph.run("CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE")
    graph.run("CREATE CONSTRAINT actor_name IF NOT EXISTS FOR (a:Actor) REQUIRE a.name IS UNIQUE")
    logger.info("Constraints created (if not exists).")

def insert_movies(df):
    data = df[["id", "title", "overview", "processed_overview", "type", "embedding"]].to_dict('records')
    query = """
    UNWIND $data AS row
    MERGE (m:Movie {id: row.id})
    SET m.title = row.title,
        m.overview = row.overview,
        m.processed_overview = row.processed_overview,
        m.type = row.type,
        m.embedding = row.embedding
    """
    graph.run(query, data=data)
    logger.info("Inserted Movie nodes with embeddings.")

def insert_genres(df):
    all_genres = set(g for genres in df["genre"] for g in genres)
    data = [{"name": g} for g in all_genres]
    query = "UNWIND $data AS row MERGE (:Genre {name: row.name})"
    graph.run(query, data=data)
    logger.info("Inserted Genre nodes.")

def insert_actors(df):
    all_actors = set(a for actors in df["actors"] for a in actors)
    data = [{"name": a} for a in all_actors]
    query = "UNWIND $data AS row MERGE (:Actor {name: row.name})"
    graph.run(query, data=data)
    logger.info("Inserted Actor nodes.")

def insert_has_genre(df):
    data = [
        {"id": row["id"], "genre": genre}
        for _, row in df.iterrows()
        for genre in row["genre"]
    ]
    query = """
    UNWIND $data AS row
    MATCH (m:Movie {id: row.id})
    MATCH (g:Genre {name: row.genre})
    MERGE (m)-[:HAS_GENRE]->(g)
    """
    graph.run(query, data=data)
    logger.info("Created HAS_GENRE relationships.")

def insert_features(df):
    data = [
        {"id": row["id"], "actor": actor}
        for _, row in df.iterrows()
        for actor in row["actors"]
    ]
    query = """
    UNWIND $data AS row
    MATCH (m:Movie {id: row.id})
    MATCH (a:Actor {name: row.actor})
    MERGE (m)-[:FEATURES]->(a)
    """
    graph.run(query, data=data)
    logger.info("Created FEATURES relationships.")

# === Main Pipeline ===

def main():
    # Set relative path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movie_path = os.path.join(base_dir, 'Dataset', 'movie.json')
    save_path = os.path.join(base_dir, 'Result', 'processed_movie_data.json')

    # Load data
    data_movies = load_data_from_json(movie_path)
    df_movies = create_dataframe(data_movies)

    df_movies["type"] = "movie"

    # Preprocess text and generate embeddings
    df_movies["processed_overview"] = df_movies["overview"].apply(preprocess_text)
    df_movies["embedding"] = generate_embeddings(df_movies["processed_overview"].tolist())

    # Create graph schema and insert data
    create_constraints()
    insert_movies(df_movies)
    insert_genres(df_movies)
    insert_actors(df_movies)
    insert_has_genre(df_movies)
    insert_features(df_movies)

    # Save processed data
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_movies.to_json(save_path, orient='records', force_ascii=False)
    logger.info(f"Processing complete. Data saved to {save_path}")

if __name__ == "__main__":
    main()
