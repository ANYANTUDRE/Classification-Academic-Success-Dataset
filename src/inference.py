import joblib
import os
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder

actual_model = "xgb"

def predict(models):
    test  = pd.read_csv(config.TEST_FILE)
    sample = pd.read_csv(config.SAMPLE_FILE)
    predictions = None

    label_encoder = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{label_encoder}.pkl"))
    test.Target = label_encoder.fit_transform(test.Target)

    for model in models:
        for fold in range(5):
            clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.pkl"))
            preds = clf.predict_proba(test)
            if fold == 0:
                predictions = preds
            else:
                predictions += preds

        predictions = predictions / 5

        if model == actual_model:
            ens_preds = predictions
        else:
            ens_preds += predictions

    ens_preds = np.argmax(ens_preds / len(models), axis=1)
    sample.Target = label_encoder.inverse_transform(ens_preds.astype(float))
    print(sample.head())
    return sample

if __name__ == "__main__":
    models = [actual_model]                #["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(models)
    submission.to_csv(f"../output/{actual_model}.csv", index=False)