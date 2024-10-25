import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set the paths
training_dir = r'D:\Machine learning\On whole preprocessed dataset(Augmented)\Whole preprocessed train dataset(augmented)'  # Path to the training dataset
validation_dir = r'D:\Machine learning\On whole preprocessed dataset(Augmented)\Whole preprocessed validation dataset(augmented)'  # Path to the validation dataset
cnn_model_save_path = r'D:\Machine learning\RESNET50 Models\Train model resnet50 train validation whole preprocessed(augmented)\Model_5.h5'  # Path to save the CNN model
class_names = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

# Define image size and batch size
image_size = (224, 224)  # Adjusted for better performance with ResNet50
batch_size = 8  # Adjust as needed
epochs = 10

# Define data generators without preprocessing
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Use 'sparse' for sparse categorical crossentropy
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights_dict = dict(enumerate(class_weights))

# Build the CNN model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)
cnn_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the CNN model
cnn_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model using the generator
cnn_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    class_weight=class_weights_dict
)

# Save the CNN model
cnn_model.save(cnn_model_save_path)

# Predict on the validation set using the CNN model
y_pred_cnn = cnn_model.predict(val_generator)
y_pred_classes_cnn = np.argmax(y_pred_cnn, axis=1)

# Print confusion matrix and classification report for CNN
print("CNN Confusion Matrix:")
cm_cnn = confusion_matrix(val_generator.classes, y_pred_classes_cnn, normalize='true')
print(cm_cnn)

print("CNN Classification Report:")
print(classification_report(val_generator.classes, y_pred_classes_cnn))

# Plot confusion matrix for CNN
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt=".2f", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('CNN Normalized Confusion Matrix')
plt.show()
