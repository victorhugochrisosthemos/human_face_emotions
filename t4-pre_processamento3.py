import os
import numpy as np
from PIL import Image

dataset_path = "dataset_pre_processado"
image_size = (48, 48)

X = []
y = []

# mapeia cada pasta para um rótulo numérico
classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}

for classe in classes:
    caminho_classe = os.path.join(dataset_path, classe)

    if not os.path.isdir(caminho_classe):
        continue

    for arquivo in os.listdir(caminho_classe):
        caminho_img = os.path.join(caminho_classe, arquivo)

        try:
            # abre imagem
            img = Image.open(caminho_img)

            # converte para tons de cinza (1 canal)
            img = img.convert("L")

            # garante tamanho 48x48
            img = img.resize(image_size)

            # converte para array numpy
            img_array = np.array(img, dtype=np.float32)

            # normaliza para [0, 1]
            img_array /= 255.0

            # adiciona canal (48, 48, 1)
            img_array = np.expand_dims(img_array, axis=-1)

            X.append(img_array)
            y.append(class_to_label[classe])

        except Exception as e:
            print(f"Erro ao processar {caminho_img}: {e}")

X = np.array(X)
y = np.array(y)

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
