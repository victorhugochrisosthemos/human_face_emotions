import os
import shutil
from PIL import Image

dataset_origem = "dataset_original"
dataset_destino = "dataset_pre_processado"

extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# cria pasta do novo dataset
os.makedirs(dataset_destino, exist_ok=True)

for pasta in os.listdir(dataset_origem):
    caminho_pasta_origem = os.path.join(dataset_origem, pasta)

    if not os.path.isdir(caminho_pasta_origem):
        continue

    caminho_pasta_destino = os.path.join(dataset_destino, pasta)
    os.makedirs(caminho_pasta_destino, exist_ok=True)

    total = 0
    mantidas = 0

    for arquivo in os.listdir(caminho_pasta_origem):
        if arquivo.lower().endswith(extensoes):
            total += 1
            caminho_imagem = os.path.join(caminho_pasta_origem, arquivo)

            try:
                with Image.open(caminho_imagem) as img:
                    if img.size == (48, 48):
                        shutil.copy2(
                            caminho_imagem,
                            os.path.join(caminho_pasta_destino, arquivo)
                        )
                        mantidas += 1
            except Exception as e:
                print(f"Erro ao processar {caminho_imagem}: {e}")

    print(f"Pasta: {pasta}")
    print(f"Imagens totais: {total}")
    print(f"Imagens mantidas (48x48): {mantidas}")
    print("-" * 40)
