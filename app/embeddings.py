"""Carregamento dos modelos e geração dos vetores de imagem e de texto (transformers, CPU)."""

from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.config import ABORDAGENS_EMBEDDING

DEVICE = "cpu"


@lru_cache(maxsize=3)
def carregar_modelo(abordagem: str):
    """Carrega (modelo, processador) da abordagem uma vez e mantém em cache."""
    nome = ABORDAGENS_EMBEDDING[abordagem]["modelo"]
    modelo = AutoModel.from_pretrained(nome).to(DEVICE).eval()
    return modelo, AutoProcessor.from_pretrained(nome)


def dimensao(abordagem: str) -> int:
    return ABORDAGENS_EMBEDDING[abordagem]["dimensao"]


def _extrair_tensor(saida) -> torch.Tensor:
    """Garante que a saída do modelo seja convertida em um torch.Tensor limpo."""
    if isinstance(saida, torch.Tensor):
        return saida
    if hasattr(saida, "image_embeds") and saida.image_embeds is not None:
        return saida.image_embeds
    if hasattr(saida, "text_embeds") and saida.text_embeds is not None:
        return saida.text_embeds
    if hasattr(saida, "pooler_output") and saida.pooler_output is not None:
        return saida.pooler_output
    return saida[0]


def _normalizar(saida) -> list[list[float]]:
    tensor = _extrair_tensor(saida)
    return torch.nn.functional.normalize(tensor, p=2, dim=-1).tolist()


def _abrir(caminho: str) -> Image.Image | None:
    try:
        with Image.open(caminho) as im:
            return im.convert("RGB")
    except Exception:
        return None


def embed_imagens(caminhos: list[str], abordagem: str) -> list[list[float] | None]:
    """Vetores normalizados de um lote; None para cada imagem ilegível."""
    imagens = [_abrir(c) for c in caminhos]
    validas = [(i, im) for i, im in enumerate(imagens) if im is not None]
    resultado: list[list[float] | None] = [None] * len(caminhos)
    if not validas:
        return resultado
    modelo, proc = carregar_modelo(abordagem)
    with torch.no_grad():
        entrada = proc(images=[im for _, im in validas], return_tensors="pt").to(DEVICE)
        saida = modelo.get_image_features(**entrada)
        vetores = _normalizar(saida)
    for (i, _), v in zip(validas, vetores):
        resultado[i] = v
    return resultado


def embed_imagem(caminho: str, abordagem: str) -> list[float]:
    v = embed_imagens([caminho], abordagem)[0]
    if v is None:
        raise ValueError(f"imagem ilegível: {caminho}")
    return v


def embed_texto(texto: str, abordagem: str) -> list[float]:
    modelo, proc = carregar_modelo(abordagem)
    kwargs = {"padding": "max_length", "max_length": 64} if "siglip" in ABORDAGENS_EMBEDDING[abordagem]["modelo"] else {"padding": True}
    with torch.no_grad():
        entrada = proc(text=[texto], return_tensors="pt", truncation=True, **kwargs).to(DEVICE)
        saida = modelo.get_text_features(**entrada)
        return _normalizar(saida)[0]


def aquecer(abordagem: str) -> None:
    """Uma inferência descartável para tirar o custo de carga das medições."""
    embed_texto("aquecimento", abordagem)