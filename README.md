# Cardiovascular Disease Prediction - ML CI/CD (Practical-5)

Dataset: cardio_train.csv (70,000 patient records, target = `cardio`).

Pipeline: train -> quality gate (accuracy >= 0.70) -> ML tests -> API tests -> release candidate artifact.

Run locally:
    pip install -r requirements.txt
    python train_model.py && python quality_gate.py
    python -m unittest test_ml_pipeline.py test_app.py -v
    python app.py
