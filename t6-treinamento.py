# ============================================================
# TREINAMENTO DA CNN — 5 BLOCOS, DATA AUGMENTATION, MATRIZ DE CONFUSÃO
# ============================================================

import os
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, Flatten,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. CARREGAR DATASET DO DISCO (já 48x48 grayscale)
# ============================================================

dataset_path = "dataset_pre_processado"
image_size = (48, 48)

X = []
y = []

classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}

print("Classes mapeadas:", class_to_label)

for classe in classes:
    caminho_classe = os.path.join(dataset_path, classe)

    if not os.path.isdir(caminho_classe):
        continue

    print(f"Carregando imagens da classe: {classe}")

    for arquivo in os.listdir(caminho_classe):
        caminho_img = os.path.join(caminho_classe, arquivo)

        try:
            img = Image.open(caminho_img).convert("L")   # grayscale
            img = img.resize(image_size)                # garante 48x48

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)

            X.append(img_array)
            y.append(class_to_label[classe])

        except Exception as e:
            print("Erro ao carregar:", caminho_img, e)

X = np.array(X)
y = np.array(y)

print("\nDataset carregado:")
print("X:", X.shape)
print("y:", y.shape)

# ============================================================
# 2. DIVISÃO TREINO / VALIDAÇÃO / TESTE
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nSplit realizado:")
print("Treino:", X_train.shape, y_train.shape)
print("Validação:", X_val.shape, y_val.shape)
print("Teste:", X_test.shape, y_test.shape)

# ============================================================
# 3. DATA AUGMENTATION (APENAS NO TREINO)
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# ============================================================
# 4. CNN COM 5 BLOCOS CONVOLUCIONAIS
# ============================================================

model = Sequential()

# BLOCO 1 — 32 filtros
model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.20))

# BLOCO 2 — 64 filtros
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOCO 3 — 128 filtros
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.30))

# BLOCO 4 — 256 filtros
model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.35))

# BLOCO 5 — 512 filtros
model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.40))

# FULLY CONNECTED
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.50))

# SAÍDA
model.add(Dense(5, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nResumo da CNN:")
model.summary()

# ============================================================
# 5. CALLBACK PARA LOGS + CSV
# ============================================================

csv_file = "training_log.csv"

with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

class CustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {logs['loss']:.4f}  // "
            f"Train Acc: {logs['accuracy']:.4f} // "
            f"Val Loss: {logs['val_loss']:.4f} // "
            f"Val Acc: {logs['val_accuracy']:.4f}"
        )

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                logs["loss"], logs["accuracy"],
                logs["val_loss"], logs["val_accuracy"]
            ])

# ============================================================
# 6. TREINAMENTO
# ============================================================

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=40,
    callbacks=[CustomLogger()],
    verbose=0
)

# ============================================================
# 7. AVALIAÇÃO NO TESTE
# ============================================================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nRESULTADOS NO TESTE:")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================================================
# 8. CLASSIFICAÇÃO + MATRIZ DE CONFUSÃO
# ============================================================

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRELATÓRIO COMPLETO (TESTE):")
print(classification_report(
    y_test,
    y_pred,
    target_names=list(classes)
))

# MATRIZ DE CONFUSÃO
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=classes, yticklabels=classes)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Teste")
plt.show()

print("\nTreinamento concluído. training_log.csv gerado.")
