import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# =========================
# CONFIGURAÇÕES
# =========================
dataset_path = "dataset_pre_processado"
model_path = "modelo_final_ferplus.h5"
image_size = (48, 48)
num_imgs = 15

# =========================
# 1. CARREGAR MODELO
# =========================
model = load_model(model_path)

# =========================
# 2. CARREGAR TODAS AS IMAGENS
# =========================
classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}
label_to_class = {v: k for k, v in class_to_label.items()}

imgs = []
trues = []

for classe in classes:
    classe_dir = os.path.join(dataset_path, classe)

    for filename in os.listdir(classe_dir):
        path_img = os.path.join(classe_dir, filename)

        img = Image.open(path_img).convert("L")
        img = img.resize(image_size, Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)

        imgs.append(arr)
        trues.append(class_to_label[classe])

imgs = np.array(imgs)
trues = np.array(trues)

# =========================
# 3. AMOSTRAS ALEATÓRIAS
# =========================
indices = np.random.choice(len(imgs), num_imgs, replace=False)
sample_imgs = imgs[indices]
sample_trues = trues[indices]

# =========================
# 4. PREVISÕES
# =========================
pred_probs = model.predict(sample_imgs)
pred_classes = np.argmax(pred_probs, axis=1)

# =========================
# 5. PLOTAR RESULTADOS
# =========================
cols = 5
rows = num_imgs // cols if num_imgs % cols == 0 else (num_imgs // cols) + 1

plt.figure(figsize=(cols * 2, rows * 2.4))

for i in range(num_imgs):
    plt.subplot(rows, cols, i + 1)

    plt.imshow(sample_imgs[i].reshape(48, 48),
               cmap="gray",
               interpolation="bilinear")  # suavização

    plt.axis("off")

    predicted = label_to_class[pred_classes[i]]
    true_label = label_to_class[sample_trues[i]]

    #color = "green" if predicted == true_label else "red"

    #plt.title(f"Prev={predicted}\nTrue={true_label}", fontsize=8, color=color)

    plt.title(f"Prev={predicted}\nTrue={true_label}", fontsize=8)

plt.tight_layout()
plt.show()
