import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os


class TitanicSurvivalModel:
    """
    Titanic Survival Prediction Model
    Predicts whether a passenger would survive based on their characteristics
    """

    def __init__(self):
        self.model = None
        self.sex_encoder = LabelEncoder()
        self.embarked_encoder = LabelEncoder()
        self.feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    def load_data(self, file_path=None):
        """
        Load and preprocess Titanic data
        Returns: X (features), y (survival labels)
        """
        if file_path is None:
            # Use absolute path
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, 'data', 'titanic.csv')
        """
        Load and preprocess Titanic data
        Returns: X (features), y (survival labels)
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✓ Loaded {len(df)} passenger records")

            # Select relevant columns
            df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

            # Handle missing values
            df['Age'].fillna(df['Age'].median(), inplace=True)
            df['Fare'].fillna(df['Fare'].median(), inplace=True)
            df['Embarked'].fillna('S', inplace=True)  # Most common port

            # Encode categorical variables
            df['Sex'] = self.sex_encoder.fit_transform(df['Sex'])
            df['Embarked'] = self.embarked_encoder.fit_transform(df['Embarked'])

            # Separate features and target
            X = df[self.feature_names]
            y = df['Survived']

            print("\nDataset Statistics:")
            print(f"  Total Passengers: {len(df)}")
            print(f"  Survivors: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
            print(f"  Non-Survivors: {len(y) - y.sum()} ({(len(y) - y.sum()) / len(y) * 100:.1f}%)")

            return X, y

        except FileNotFoundError:
            print(f"✗ Error: {file_path} not found")
            return None, None

    def train(self, X, y):
        """
        Train the survival prediction model
        Uses Random Forest Classifier
        """
        print("\n--- Training Model ---")

        # Split data: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5
        )
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_predictions = self.model.predict(X_train)
        test_predictions = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        print(f"\nModel Performance:")
        print(f"  Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"  Testing Accuracy: {test_accuracy * 100:.2f}%")

        # Detailed classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_predictions,
                                    target_names=['Did Not Survive', 'Survived']))

        # Feature importance
        self._show_feature_importance()

        return test_accuracy

    def _show_feature_importance(self):
        """Display which features matter most for survival prediction"""
        if self.model:
            importance = self.model.feature_importances_
            print("\nFeature Importance (What matters most for survival):")
            feature_imp = sorted(zip(self.feature_names, importance),
                                 key=lambda x: x[1], reverse=True)
            for name, imp in feature_imp:
                print(f"  {name}: {imp:.4f}")

    def predict(self, pclass, sex, age, sibsp, parch, fare, embarked):
        """
        Predict survival probability for a passenger

        Parameters:
        - pclass: Passenger class (1, 2, or 3)
        - sex: 'male' or 'female'
        - age: Age in years
        - sibsp: Number of siblings/spouses aboard
        - parch: Number of parents/children aboard
        - fare: Ticket fare
        - embarked: Port of embarkation ('S', 'C', or 'Q')

        Returns: (prediction, probability)
        - prediction: 0 (died) or 1 (survived)
        - probability: Probability of survival (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Encode categorical variables
        sex_encoded = self.sex_encoder.transform([sex])[0]
        embarked_encoded = self.embarked_encoder.transform([embarked])[0]

        # Create feature array
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        return prediction, probability[1]  # Return survival probability

    def save_model(self, model_dir='model_files'):
        """Save trained model and encoders to disk"""
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'titanic_model.pkl')
        sex_encoder_path = os.path.join(model_dir, 'sex_encoder.pkl')
        embarked_encoder_path = os.path.join(model_dir, 'embarked_encoder.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.sex_encoder, sex_encoder_path)
        joblib.dump(self.embarked_encoder, embarked_encoder_path)

        print(f"\n✓ Model saved to {model_path}")
        print(f"✓ Encoders saved to {model_dir}")

    def load_model(self, model_dir='model_files'):
        """Load pre-trained model and encoders from disk"""
        model_path = os.path.join(model_dir, 'titanic_model.pkl')
        sex_encoder_path = os.path.join(model_dir, 'sex_encoder.pkl')
        embarked_encoder_path = os.path.join(model_dir, 'embarked_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.sex_encoder = joblib.load(sex_encoder_path)
            self.embarked_encoder = joblib.load(embarked_encoder_path)
            print(f"✓ Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"✗ Model files not found in {model_dir}")
            return False


def train_and_save_model():
    """
    Main training function - run this to create the model
    """
    print("=" * 60)
    print("TITANIC SURVIVAL PREDICTION MODEL TRAINING")
    print("=" * 60)

    # Initialize model
    titanic_model = TitanicSurvivalModel()

    # Load data
    X, y = titanic_model.load_data()
    if X is None:
        return

    # Train model
    titanic_model.train(X, y)

    # Save trained model
    titanic_model.save_model()

    # Test predictions
    print("\n--- Testing Sample Predictions ---")

    # Example 1: First-class female
    pred, prob = titanic_model.predict(
        pclass=1, sex='female', age=30, sibsp=0, parch=0, fare=100, embarked='S'
    )
    print(f"1st Class Female, Age 30: {'Survived ✓' if pred == 1 else 'Did Not Survive ✗'} "
          f"(Probability: {prob * 100:.1f}%)")

    # Example 2: Third-class male
    pred, prob = titanic_model.predict(
        pclass=3, sex='male', age=25, sibsp=0, parch=0, fare=7.5, embarked='S'
    )
    print(f"3rd Class Male, Age 25: {'Survived ✓' if pred == 1 else 'Did Not Survive ✗'} "
          f"(Probability: {prob * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()