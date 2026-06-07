import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .clustering import RFM_FEATURES


def train_random_forest(
    rfm_labeled: pd.DataFrame,
    target_col: str = "Segment",
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = rfm_labeled[RFM_FEATURES]
    y = rfm_labeled[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=random_state, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return model, report


def predict_segment(model, recency: float, frequency: float, monetary: float) -> str:
    X_new = pd.DataFrame(
        [[recency, frequency, monetary]], columns=RFM_FEATURES
    )
    return str(model.predict(X_new)[0])
