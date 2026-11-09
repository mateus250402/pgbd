"""Conexão com o PostgreSQL e criação das extensões."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)


def conexao() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
    register_vector(conn)
    return conn


def criar_extensoes(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def criar_schema() -> None:
    """Aplica schema.sql (idempotente)."""
    from pathlib import Path

    sql = (Path(__file__).resolve().parent.parent / "schema.sql").read_text()
    with conexao() as conn:
        conn.execute(sql)
