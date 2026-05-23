"""
Re-evaluate all saved models on the test dataset and update DB metrics.
Run from project root: python re_evaluate.py
"""
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Setup Flask app context so we can access the database
sys.path.insert(0, os.path.dirname(__file__))
from app import create_app
from app.models import db, TrainedModel
from models.deep_learning import DataLoader

app = create_app()

CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

with app.app_context():
    # Load test data once
    dataset_folder = os.path.join(os.path.dirname(__file__), 'Dataset')
    print(f"[1/3] Loading test dataset from: {dataset_folder}")
    data_loader = DataLoader(dataset_folder)
    _, _, X_test, y_test = data_loader.load_train_test_data()
    print(f"      Loaded {len(X_test)} test images.\n")

    # Get all models
    models = TrainedModel.query.all()
    if not models:
        print("No trained models found in database.")
        sys.exit(0)

    print(f"[2/3] Found {len(models)} model(s) to re-evaluate.\n")

    for model_obj in models:
        print(f"{'='*60}")
        print(f"  Model: {model_obj.model_name}")
        print(f"  Algorithm: {model_obj.algorithm}")
        print(f"  File: {model_obj.model_path}")

        if not os.path.exists(model_obj.model_path):
            print(f"  ❌ SKIPPED — model file not found!\n")
            continue

        # Load and predict
        from tensorflow import keras
        print(f"  Loading model...")
        keras_model = keras.models.load_model(model_obj.model_path)

        print(f"  Running predictions on {len(X_test)} test images...")
        predictions = keras_model.predict(X_test, verbose=0)
        pred_labels = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, pred_labels)
        cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2, 3])
        report = classification_report(
            y_test, pred_labels,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Print results
        print(f"\n  --- RESULTS ---")
        print(f"  Accuracy:  {accuracy:.4f}  ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Confusion Matrix:")
        for i, row in enumerate(cm.tolist()):
            print(f"    {CLASS_NAMES[i]:>20s}: {row}")

        # Update database
        model_obj.accuracy = float(accuracy)
        model_obj.precision = float(precision)
        model_obj.recall = float(recall)
        model_obj.f1_score = float(f1)
        model_obj.set_confusion_matrix({'matrix': cm.tolist(), 'classes': CLASS_NAMES})

        print(f"\n  ✅ Database updated.\n")

    db.session.commit()
    print(f"{'='*60}")
    print(f"[3/3] All models re-evaluated and database committed successfully!")
    print(f"      Refresh the admin portal to see updated metrics.")
