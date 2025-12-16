import os
import random

dataset_path = "dataset_pre_processado"
limite_por_pasta = 8003

extensoes = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

for pasta in os.listdir(dataset_path):
    caminho_pasta = os.path.join(dataset_path, pasta)

    if not os.path.isdir(caminho_pasta):
        continue

    imagens = [
        img for img in os.listdir(caminho_pasta)
        if img.lower().endswith(extensoes)
    ]

    total = len(imagens)

    if total <= limite_por_pasta:
        print(f"Pasta: {pasta}")
        print(f"Total de imagens: {total} (nenhuma removida)")
        print("-" * 40)
        continue

    # embaralha para remover aleatoriamente
    random.shuffle(imagens)

    # imagens que serão removidas
    excedentes = imagens[limite_por_pasta:]

    for img in excedentes:
        os.remove(os.path.join(caminho_pasta, img))

    print(f"Pasta: {pasta}")
    print(f"Total original: {total}")
    print(f"Imagens removidas: {len(excedentes)}")
    print(f"Total final: {limite_por_pasta}")
    print("-" * 40)
