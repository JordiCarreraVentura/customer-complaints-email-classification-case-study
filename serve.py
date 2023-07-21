import os

# Set environment variable for the tracking URL where the Model Registry resides
os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"

# Serve the production model from the model registry
# {run_name}-001
# char_vectorizer_noidf.reducer_2000.selector_svm.classifier_svm-001

cmd = 'mlflow models serve -m "models:/char_vectorizer_noidf.reducer_2000.selector_svm.classifier_svm-001/5"'

os.system(cmd)
