import os
import sys
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    RecommendInput,
)



# =====================================================================
# 1. Configuración de Conexión (compatible con local y Docker Compose)
# =====================================================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "catalogo_libros"
MODEL_NAME = "intfloat/multilingual-e5-small"

print(f"[*] Conectando a Qdrant en http://{QDRANT_HOST}:{QDRANT_PORT}...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# =====================================================================
# 2. Carga del Modelo de Embeddings
# =====================================================================
print(f"[*] Cargando modelo de embeddings ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)
vector_size = model.get_sentence_embedding_dimension()

# =====================================================================
# 3. Preparación de Colección e Índices de Payload
# =====================================================================
def preparar_coleccion():
    """Crea la colección e índices si no existen en Qdrant."""
    if not client.collection_exists(COLLECTION_NAME):
        print(f"[*] Creando coleccion '{COLLECTION_NAME}' (dimension: {vector_size}, distancia: Coseno)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        print("[*] Configurando indices de filtrado en el payload...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categories",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="average_rating",
            field_schema=PayloadSchemaType.FLOAT,
        )
        ingestar_datos()
    else:
        info = client.get_collection(COLLECTION_NAME)
        print(f"[*] Coleccion existente detectada con {info.points_count} libros ya almacenados en el volumen de Docker.")

# =====================================================================
# 4. Ingesta por Lotes (Dataset -> Embeddings -> Qdrant)
# =====================================================================
def ingestar_datos():
    """Descarga el dataset, calcula embeddings en lotes e inserta los puntos."""
    print("[*] Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
    csv_path = os.path.join(path, "books.csv")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    total_libros = len(df)
    batch_size = 64

    print(f"[*] Insertando {total_libros} libros en lotes de {batch_size}...")
    for start_idx in range(0, total_libros, batch_size):
        end_idx = min(start_idx + batch_size, total_libros)
        batch_df = df.iloc[start_idx:end_idx]

        descriptions = batch_df["description"].tolist()
        embeddings = model.encode(descriptions, show_progress_bar=False)

        points = []
        for point_id, (_, row), vector in zip(batch_df.index, batch_df.iterrows(), embeddings):
            payload = {
                "title": str(row.get("title", "")),
                "description": str(row.get("description", "")),
                "categories": str(row.get("categories", "")),
                "average_rating": float(row.get("average_rating", 0.0)) if pd.notnull(row.get("average_rating")) else 0.0,
                "authors": str(row.get("authors", "")),
            }
            points.append(
                PointStruct(
                    id=int(point_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"    Progreso: {end_idx}/{total_libros}", end="\r")

    print("\n[*] Ingesta completada con exito.")

# =====================================================================
# 5. Funciones de Búsqueda Semántica y Recomendación
# =====================================================================
def buscar_libros(query: str, genero: str = None, limit: int = 5) -> pd.DataFrame:
    """Busca libros usando el vector de la consulta y filtros opcionales."""
    query_vector = model.encode(query).tolist()
    query_filter = None

    if genero:
        query_filter = Filter(
            must=[FieldCondition(key="categories", match=MatchValue(value=genero))]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
    )

    resultados = []
    for hit in response.points:
        resultados.append({
            "id": hit.id,
            "score": round(hit.score, 4),
            "title": hit.payload.get("title"),
            "categories": hit.payload.get("categories"),
            "rating": hit.payload.get("average_rating"),
            "description": hit.payload.get("description")[:120] + "...",
        })
    return pd.DataFrame(resultados)


def recomendar_libros(me_gustan_ids: list, no_me_gustan_ids: list = None, limit: int = 5) -> pd.DataFrame:
    """Recomienda mediante álgebra vectorial en Qdrant (positivos vs negativos)."""
    if no_me_gustan_ids is None:
        no_me_gustan_ids = []

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=models.RecommendQuery(
            recommend=models.RecommendInput(
                positive=me_gustan_ids,
                negative=no_me_gustan_ids,
            )
        ),
        limit=limit,
    )

    resultados = []
    for hit in response.points:
        resultados.append({
            "id": hit.id,
            "score": round(hit.score, 4),
            "title": hit.payload.get("title"),
            "categories": hit.payload.get("categories"),
            "rating": hit.payload.get("average_rating"),
            "description": hit.payload.get("description")[:120] + "...",
        })
    return pd.DataFrame(resultados)

# =====================================================================
# 6. Punto de Entrada Principal
# =====================================================================
if __name__ == "__main__":
    preparar_coleccion()

    print("\n" + "=" * 60)
    print("DEMOSTRACION: BUSQUEDA SEMANTICA Y RECOMENDADOR")
    print("=" * 60)

    # 1. Búsqueda Semántica de prueba
    consulta = "space exploration alien contact science fiction"
    print(f"\n1. Ejecutando busqueda semantica para: '{consulta}'")
    df_busqueda = buscar_libros(query=consulta, limit=3)
    print(df_busqueda[["id", "score", "title", "categories", "rating"]].to_string(index=False))

    # 2. Recomendación de prueba
    if not df_busqueda.empty:
        id_positivo = int(df_busqueda.iloc[0]["id"])
        print(f"\n2. Generando recomendaciones basadas en el libro ID {id_positivo} ('{df_busqueda.iloc[0]['title']}'):")
        df_recs = recomendar_libros(me_gustan_ids=[id_positivo], limit=3)
        print(df_recs[["id", "score", "title", "categories", "rating"]].to_string(index=False))

    print("\n[+] Proceso finalizado correctamente.")