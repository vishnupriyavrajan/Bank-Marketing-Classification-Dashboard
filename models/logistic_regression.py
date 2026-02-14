from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_logistic_regression(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
