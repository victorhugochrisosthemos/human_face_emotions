import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# ================
# CONFIGURAÇÕES
# ================

dataset_path = "dataset_pre_processado"   # onde estão as imagens organizadas por pasta
model_path = "modelo_final_ferplus.h5"    # seu modelo já treinado
image_size = (48, 48)
num_imgs = 10   # quantas imagens mostrar

# ================
# 1. CARREGA MODELO
# ================
model = load_model(model_path)

# ================
# 2. LER IMAGENS DO DATASET DE TESTE
# (pegaremos aleatoriamente de todas as classes)
# ================

classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}
label_to_class = {v: k for k, v in class_to_label.items()}

# Carregar todas as imagens do dataset
imgs = []
trues = []

for classe in classes:
    classe_dir = os.path.join(dataset_path, classe)
    for filename in os.listdir(classe_dir):
        path_img = os.path.join(classe_dir, filename)

        # abre imagem
        img = Image.open(path_img).convert("L")
        img = img.resize(image_size)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)

        imgs.append(arr)
        trues.append(class_to_label[classe])

imgs = np.array(imgs)
trues = np.array(trues)

# ================
# 3. SELECIONAR AMOSTRAS ALEATÓRIAS
# ================
indices = np.random.choice(len(imgs), num_imgs, replace=False)
sample_imgs = imgs[indices]
sample_trues = trues[indices]

# ================
# 4. FAZER PREVISÕES
# ================
pred_probs = model.predict(sample_imgs)
pred_classes = np.argmax(pred_probs, axis=1)

# ================
# 5. PLOTAR RESULTADOS
# ================
plt.figure(figsize=(16, 6))

for i in range(num_imgs):
    plt.subplot(2, num_imgs//2, i+1)
    plt.imshow(sample_imgs[i].reshape(48, 48), cmap="gray")
    plt.axis("off")

    predicted = label_to_class[pred_classes[i]]
    true_label = label_to_class[sample_trues[i]]

    plt.title(f"Previsão = {predicted}\nTrue = {true_label}", fontsize=10)

plt.tight_layout()
plt.show()
