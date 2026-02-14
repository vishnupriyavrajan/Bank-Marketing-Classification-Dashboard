from xgboost import XGBClassifier
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_xgboost(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
