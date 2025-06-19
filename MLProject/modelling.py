import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import joblib

# Atur URI tracking ke local MLflow server (pastikan mlflow ui sedang aktif di 127.0.0.1:5000)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Buat atau gunakan experiment yang sudah ada
mlflow.set_experiment("Diabetes Classification")

# Aktifkan autolog untuk mencatat otomatis (opsional, bisa diganti manual logging)
mlflow.sklearn.autolog()

# Load dataset hasil preprocessing
data = pd.read_csv("../preprocessing/preprocessing/diabetes_preprocessing.csv")

# Pisahkan fitur dan target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Pastikan target bertipe integer (klasifikasi, bukan continuous)
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Contoh input untuk input_example di log_model
input_example = X_train.iloc[0:5]

# Mulai pencatatan eksperimen dengan MLflow
with mlflow.start_run() as run:

    # Parameter model
    n_estimators = 100
    max_depth = 10

    # Logging parameter manual (meskipun autolog aktif, ini untuk jaga-jaga)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Inisialisasi dan latih model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluasi akurasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # Logging model secara eksplisit (lebih aman dan fleksibel)
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=input_example
    )
    # Save model secara manual ke .pkl
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.pkl"
    joblib.dump(clf, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log visualisasi confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    fig_path = "model/training_confusion_matrix.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path, artifact_path="model")

    # Log environment dependencies
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            f'python=3.12',
            'scikit-learn=1.5.2',
            'pandas=2.2.3',
            'numpy=1.26.4',
            'pip',
            {
                'pip': ['mlflow==2.18.0']
            }
        ],
        'name': 'mlflow-env'
    }

    import yaml
    with open("model/conda.yaml", "w") as f:
        yaml.dump(conda_env, f)
    mlflow.log_artifact("model/conda.yaml", artifact_path="model")

    # Logging model via MLflow (resmi)
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None  # bisa diisi kalau ingin daftar model
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Akurasi Model: {acc:.4f}")


    print(f"Akurasi Model: {acc:.4f}")
