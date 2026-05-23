import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import json
from datetime import datetime
import logging

def log_data_loading_errors():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Load and preprocess MRI images"""
    
    def __init__(self, dataset_path, img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    def load_images(self, folder_path):
        """Load images from folder"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(folder_path, class_name)
            if not os.path.exists(class_folder):
                logging.warning(f"Warning: {class_folder} not found")
                continue
            
            image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for image_file in image_files:
                try:
                    img_path = os.path.join(class_folder, image_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        # Resize image
                        img = cv2.resize(img, self.img_size)
                        # Convert to 3 channels for ResNet
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        # Normalize
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(class_idx)
                    else:
                        logging.error(f"Error: Image is None at {img_path}")
                except Exception as e:
                    logging.error(f"Error loading {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def load_train_test_data(self):
        """Load train and test data"""
        train_path = os.path.join(self.dataset_path, 'train')
        test_path = os.path.join(self.dataset_path, 'test')
        
        print("Loading training data...")
        X_train, y_train = self.load_images(train_path)
        
        print("Loading test data...")
        X_test, y_test = self.load_images(test_path)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test

class ResNet101Model:
    """ResNet-101 Transfer Learning Model"""
    
    def __init__(self, num_classes=4, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build ResNet-101 model with transfer learning"""
        # Load pre-trained ResNet101
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(self.img_size[0], self.img_size[1], 3))
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Build new model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Data augmentation
        data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Train model
        self.history = self.model.fit(
            data_gen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        predictions = self.model.predict(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        accuracy = accuracy_score(y_test, pred_labels)
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2, 3])
        report = classification_report(
            y_test, pred_labels,
            labels=[0, 1, 2, 3],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def save_model(self, path):
        """Save model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model"""
        verify_model_path(path)
        self.model = keras.models.load_model(path)
        logging.debug(f'Model loaded from {path}')

def verify_model_path(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path does not exist: {model_path}')
    logging.debug(f'Model path verified: {model_path}')

class UNetModel:
    """U-Net Segmentation Model for feature extraction"""
    
    def __init__(self, num_classes=4, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build U-Net inspired model"""
        inputs = keras.Input(shape=(self.img_size[0], self.img_size[1], 3))
        
        # Encoder
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
        
        # Global Average Pooling for classification
        gap = layers.GlobalAveragePooling2D()(c4)
        
        # Classification head
        dense1 = layers.Dense(128, activation='relu')(gap)
        dense1 = layers.Dropout(0.5)(dense1)
        dense2 = layers.Dense(64, activation='relu')(dense1)
        dense2 = layers.Dropout(0.3)(dense2)
        outputs = layers.Dense(self.num_classes, activation='softmax')(dense2)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        self.history = self.model.fit(
            data_gen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        predictions = self.model.predict(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        accuracy = accuracy_score(y_test, pred_labels)
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2, 3])
        report = classification_report(
            y_test, pred_labels,
            labels=[0, 1, 2, 3],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def save_model(self, path):
        """Save model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)

class SimpleDLModel:
    """Simple Deep Learning Model - CNN"""
    
    def __init__(self, num_classes=4, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build simple CNN model"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        self.history = self.model.fit(
            data_gen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        predictions = self.model.predict(X_test)
        pred_labels = np.argmax(predictions, axis=1)
        
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        accuracy = accuracy_score(y_test, pred_labels)
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2, 3])
        report = classification_report(
            y_test, pred_labels,
            labels=[0, 1, 2, 3],
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    def save_model(self, path):
        """Save model"""
        self.model.save(path)
    
    def load_model(self, path):
        """Load model"""
        self.model = keras.models.load_model(path)
