from wandb.apis.importers.mlflow import MlflowRun
from wandb.apis.importers.mlflow import MlflowImporter
import mlflow

def main() -> None:

    print(mlflow.get_tracking_uri())

    importer = MlflowImporter(mlflow_tracking_uri="...")

    runs = importer.collect_runs()
    importer.import_runs(runs)

if __name__=="__main__":
    main()
    