import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing.encoder import CategoricalEncoder
from src.features.selector import FeatureSelector
from src.features.transformer import FeatureTransformer
from src.models.trainer import ModelTrainer


class Pipeline:

    def __init__(self, dataset_path, model_type="random_forest", params=None):

        self.dataset_path = dataset_path
        self.model_type = model_type
        self.params = params or {}

    def execute(self):

        print("[Pipeline] Stage 1/5 → Loading Dataset")

        df = pd.read_csv(self.dataset_path)

        print(f"[Pipeline] Dataset shape: {df.shape}")

        target_col = "Survived" if "Survived" in df.columns else df.columns[-1]

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # ------------------------------------------------
        # Train/Test Split
        # ------------------------------------------------

        print("[Pipeline] Stage 2/5 → Train/Test Split")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(y.unique()) < 20 else None
        )

        # ------------------------------------------------
        # Feature Engineering
        # ------------------------------------------------

        print("[Pipeline] Stage 3/5 → Feature Engineering")

        if "Cabin" in X_train.columns:
            X_train = X_train.drop(columns=["Cabin"])
            X_test = X_test.drop(columns=["Cabin"])

        encoder = CategoricalEncoder()

        X_train = encoder.fit_transform(X_train)
        X_test = encoder.transform(X_test)

        selector = FeatureSelector(method="variance")

        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

        transformer = FeatureTransformer(polynomial_degree=1)

        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

        # ------------------------------------------------
        # Model Training
        # ------------------------------------------------

        print("[Pipeline] Stage 4/5 → Model Training")

        trainer = ModelTrainer(self.model_type, self.params)

        start = time.time()

        trainer.fit(X_train, y_train)

        training_time = round(time.time() - start, 3)

        print(f"[Pipeline] Training completed in {training_time}s")

        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------

        print("[Pipeline] Stage 5/5 → Evaluation")

        y_pred = trainer.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print()
        print("=" * 50)
        print("MODEL PERFORMANCE")
        print("=" * 50)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print("=" * 50)

        return {
            "accuracy": accuracy,
            "f1": f1,
            "training_time": training_time,
            "model": self.model_type
        }