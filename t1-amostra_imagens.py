import os
from PIL import Image
from collections import defaultdict

dataset_path = "dataset_original"
extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for pasta in os.listdir(dataset_path):
    caminho_pasta = os.path.join(dataset_path, pasta)

    if not os.path.isdir(caminho_pasta):
        continue

    contagem_tamanhos = defaultdict(int)
    total = 0

    for arquivo in os.listdir(caminho_pasta):
        if arquivo.lower().endswith(extensoes):
            caminho_imagem = os.path.join(caminho_pasta, arquivo)
            try:
                with Image.open(caminho_imagem) as img:
                    contagem_tamanhos[img.size] += 1
                    total += 1
            except Exception as e:
                print(f"Erro ao abrir {caminho_imagem}: {e}")

    print(f"Pasta: {pasta}")
    print(f"Quantidade de imagens: {total}")
    print("Distribuição de tamanhos:")
    for tamanho, qtd in contagem_tamanhos.items():
        print(f"  {tamanho}: {qtd} imagens")
    print("-" * 40)
