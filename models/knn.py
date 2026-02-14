from sklearn.neighbors import KNeighborsClassifier
from preprocessing import preprocess_data
from metrics import evaluate_model

def run_knn(df):
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)

    return evaluate_model(model, X_test, y_test)
