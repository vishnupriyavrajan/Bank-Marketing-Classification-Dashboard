from sklearn.naive_bayes import GaussianNB
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_naive_bayes(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = GaussianNB()
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
