# ============================================================
# CNN FER+ INSPIRED — COM BLOCOS RESIDUAIS (SEM SALVAR A CADA ÉPOCA)
# + SALVAMENTO DE LOGS EM CSV
# ============================================================

import os
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    Add, Input, GlobalAveragePooling2D, Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. CARREGAR DATASET
# ============================================================

dataset_path = "dataset_pre_processado"
image_size = (48, 48)

X, y = [], []

classes = sorted(os.listdir(dataset_path))
class_to_label = {classe: i for i, classe in enumerate(classes)}

print("Classes mapeadas:", class_to_label)

for classe in classes:
    caminho_classe = os.path.join(dataset_path, classe)

    for arquivo in os.listdir(caminho_classe):
        caminho_img = os.path.join(caminho_classe, arquivo)

        img = Image.open(caminho_img).convert("L")
        img = img.resize(image_size)

        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)

        X.append(img)
        y.append(class_to_label[classe])

X = np.array(X)
y = np.array(y)

# ============================================================
# 2. SPLIT
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

# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True
)

datagen.fit(X_train)

# ============================================================
# 4. BLOCO RESIDUAL
# ============================================================

def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# ============================================================
# 5. MODELO FER+ INSPIRED
# ============================================================

inp = Input(shape=(48,48,1))

x = Conv2D(64, (3,3), padding='same')(inp)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = residual_block(x, 64)
x = residual_block(x, 64)
x = residual_block(x, 64)

x = Conv2D(128, (3,3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = residual_block(x, 128)
x = residual_block(x, 128)

x = Conv2D(256, (3,3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = residual_block(x, 256)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)

out = Dense(5, activation="softmax")(x)

model = Model(inputs=inp, outputs=out)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 6. TREINAMENTO — SALVAR LOG EM CSV
# ============================================================

csv_logger = CSVLogger("log_treinamento.csv", separator=",", append=False)

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    verbose=1,
    callbacks=[csv_logger]
)

# ============================================================
# 7. SALVAR MODELO FINAL
# ============================================================

model.save("modelo_final_ferplus.h5")
print("\nModelo final salvo como modelo_final_ferplus.h5")
print("Log salvo como log_treinamento.csv")

# ============================================================
# 8. AVALIAÇÃO FINAL
# ============================================================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nRESULTADOS NO TESTE:")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================================================
# 9. MATRIZ DE CONFUSÃO
# ============================================================

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nRELATÓRIO COMPLETO:")
print(classification_report(
    y_test,
    y_pred,
    target_names=list(classes)
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão — FER+ Inspired CNN")
plt.show()
