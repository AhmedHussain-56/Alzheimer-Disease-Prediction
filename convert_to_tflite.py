import os
import tensorflow as tf

def convert_h5_to_tflite(h5_path, tflite_path):
    print(f"Loading {h5_path}...")
    model = tf.keras.models.load_model(h5_path)
    
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # We can also apply default optimizations to reduce the model size on disk (quantization)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    print(f"Saving TFLite model to {tflite_path}...")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("Conversion complete!\n")

if __name__ == "__main__":
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    
    # Model 1
    model1_h5 = os.path.join(saved_models_dir, "balanced_simple_20260124_124125.h5")
    model1_tflite = os.path.join(saved_models_dir, "balanced_simple_20260124_124125.tflite")
    if os.path.exists(model1_h5):
        convert_h5_to_tflite(model1_h5, model1_tflite)
    
    # Model 2
    model2_h5 = os.path.join(saved_models_dir, "resnet101_model_20260121_215729.h5")
    model2_tflite = os.path.join(saved_models_dir, "resnet101_model_20260121_215729.tflite")
    if os.path.exists(model2_h5):
        convert_h5_to_tflite(model2_h5, model2_tflite)
