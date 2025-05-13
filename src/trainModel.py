from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

def train_and_save_model(dataset_path, save_path="model/emotion_model.h5"):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical',
        color_mode='rgb',
        subset='validation'
    )

    base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(train_gen.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False  # opcional: congelar para fine-tuning leve

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[EarlyStopping(patience=5)]
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Modelo salvo em {save_path}")
