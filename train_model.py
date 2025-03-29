import mlflow
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wine_Quality_Classification")


def train_and_log_model(n_estimators, max_depth):
    wine = load_wine()
    X = wine.data
    y = wine.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)


        mlflow.log_metric("accuracy", accuracy)
        for class_name in report:
            if class_name.isdigit():
                for metric, value in report[class_name].items():
                    mlflow.log_metric(f"class_{class_name}_{metric}", value)

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(scaler, "scaler")

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, "WineApp")

        client = MlflowClient()
        client.set_model_version_tag(
            name="WineApp",
            version=registered_model.version,
            key="accuracy",
            value=str(round(accuracy, 3))
        )


def promote_best_model(model_name="WineApp"):
    client = MlflowClient()
    best_accuracy = 0
    best_version = None

    for version in client.search_model_versions(f"name='{model_name}'"):
        acc_tag = version.tags.get("accuracy")
        if acc_tag:
            current_acc = float(acc_tag)
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_version = version


    if best_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_version.version,
            stage="Production"
        )
        print(
            f" model {model_name} version {best_version.version} toproduction with accuracy {best_accuracy:.2f}")
    


if __name__ == "__main__":
    params = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 300, "max_depth": 10}
    ]

    for param in params:
        train_and_log_model(**param)


    promote_best_model()
