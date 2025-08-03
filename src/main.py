from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Step 1: Ingest data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Transform data
    transformation = DataTransformation()
    train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Step 3: Train model and get scores
    trainer = ModelTrainer()
    acc, f1 = trainer.initiate_model_trainer(train_array, test_array, preprocessor_path)

    # Step 4: Print scores
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")