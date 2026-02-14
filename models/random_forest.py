from sklearn.ensemble import RandomForestClassifier
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_random_forest(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
