from sklearn.tree import DecisionTreeClassifier
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_decision_tree(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
