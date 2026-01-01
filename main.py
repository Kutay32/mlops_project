from src.pipeline import ChurnPipeline
import os
import sys

def main():
    print("Running MLOps Project Pipeline...")
    # Use URL or local file if downloaded
    url = "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
    
    try:
        pipeline = ChurnPipeline(url)
        df = pipeline.load_data()
        pipeline.build_pipeline()
        pipeline.train(df)
        
        model_path = os.path.join(os.getcwd(), "model_artifact.joblib")
        pipeline.save(model_path)
        
        print("\nPipeline execution successful.")
        print("\nTo start the serving API, run:")
        print("uvicorn src.serving.api:app --reload")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
