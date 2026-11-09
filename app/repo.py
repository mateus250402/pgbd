"""Tabelas de embedding por modelo, preparação ao vivo e configuração."""

import math
import time
from collections.abc import Callable
from datetime import datetime, timezone

from app.config import ABORDAGENS_EMBEDDING, OPCLASS_POR_METRICA, TAMANHO_LOTE
from app.db import conexao
from app.embeddings import embed_imagens

_SUFIXO = {"Cosseno": "cos", "L2": "l2", "Produto interno": "ip"}


def _criar_indices(conn, tabela: str, total: int) -> None:
    lists = max(1, round(math.sqrt(total)))
    conn.execute("SET maintenance_work_mem = '256MB'")
    for metrica, opclass in OPCLASS_POR_METRICA.items():
        s = _SUFIXO[metrica]
        conn.execute(f"CREATE INDEX {tabela}_hnsw_{s} ON {tabela} USING hnsw (vetor {opclass})")
        conn.execute(
            f"CREATE INDEX {tabela}_ivf_{s} ON {tabela} USING ivfflat (vetor {opclass}) WITH (lists = {lists})"
        )


def preparar_modelo(abordagem: str, progresso: Callable[[int, int], None] | None = None) -> dict:
    """Recria a tabela do modelo, gera os vetores em lotes e cria os 6 índices."""
    cfg = ABORDAGENS_EMBEDDING[abordagem]
    tabela, dim = cfg["tabela"], cfg["dimensao"]
    with conexao() as conn:
        imagens = conn.execute("SELECT id, caminho FROM imagem ORDER BY id").fetchall()
        if not imagens:
            raise ValueError("Não há imagens no catálogo: coloque arquivos em ./imagens")
        conn.execute(f"DROP TABLE IF EXISTS {tabela}")
        conn.execute("DELETE FROM config_embedding WHERE abordagem = %s", (abordagem,))
        conn.execute(
            f"""CREATE TABLE {tabela} (
                  imagem_id INT PRIMARY KEY REFERENCES imagem(id) ON DELETE CASCADE,
                  vetor     vector({dim}) NOT NULL)"""
        )
        try:
            total, puladas = len(imagens), 0
            inseridas = 0
            inicio = time.perf_counter()
            for ini in range(0, total, TAMANHO_LOTE):
                lote = imagens[ini:ini + TAMANHO_LOTE]
                vetores = embed_imagens([c for _, c in lote], abordagem)
                for (img_id, _), v in zip(lote, vetores):
                    if v is None:
                        puladas += 1
                        continue
                    conn.execute(f"INSERT INTO {tabela} (imagem_id, vetor) VALUES (%s, %s)", (img_id, v))
                    inseridas += 1
                if progresso:
                    progresso(min(ini + TAMANHO_LOTE, total), total)
            tempo = time.perf_counter() - inicio
            if inseridas == 0:
                raise ValueError("Nenhuma imagem pôde ser lida")
            _criar_indices(conn, tabela, inseridas)
            conn.execute(
                """INSERT INTO config_embedding (abordagem, tabela, dimensao, total, tempo_geracao_s, gerado_em)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (abordagem, tabela, dim, inseridas, tempo, datetime.now(timezone.utc)),
            )
        except BaseException:
            conn.execute(f"DROP TABLE IF EXISTS {tabela}")
            raise
    return {"total": inseridas, "puladas": puladas, "tempo_geracao_s": tempo}


def modelo_preparado(abordagem: str) -> bool:
    return obter_config(abordagem) is not None


def obter_config(abordagem: str) -> dict | None:
    with conexao() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT abordagem, tabela, dimensao, total, tempo_geracao_s, gerado_em
                   FROM config_embedding WHERE abordagem = %s""",
                (abordagem,),
            )
            linha = cur.fetchone()
            if linha is None:
                return None
            colunas = [d.name for d in cur.description]
            return dict(zip(colunas, linha))


def total_imagens() -> int:
    with conexao() as conn:
        return conn.execute("SELECT count(*) FROM imagem").fetchone()[0]


def listar_imagens() -> list[dict]:
    """Retorna todas as imagens cadastradas para exibição inicial no grid."""
    with conexao() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, nome, caminho FROM imagem ORDER BY id")
            linhas = cur.fetchall()
            return [{"id": row[0], "nome": row[1], "caminho": row[2]} for row in linhas]