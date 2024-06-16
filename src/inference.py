import joblib
import os
import pandas as pd
import numpy as np
import config
from sklearn.preprocessing import LabelEncoder

def predict(models):
    test  = pd.read_csv(config.TEST_FILE)
    sample = pd.read_csv(config.SAMPLE_FILE)
    predictions = None

    label_encoder = LabelEncoder()
    test.Target = label_encoder.fit_transform(test.Target)

    for model in models:
        for fold in range(5):
            clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.pkl"))
            preds = clf.predict_proba(test)
            if fold == 0:
                predictions = np.argmax(preds, axis=1)
            else:
                predictions += np.argmax(preds, axis=1)

        predictions = predictions / 5

        if model == 'xgb':
            ens_preds = predictions
        else:
            ens_preds += predictions

    ens_preds = ens_preds / len(models)

    sample.Target = label_encoder.inverse_transform(ens_preds.astype(float))
    print(sample.head())
    return sample


if __name__ == "__main__":
    models = ["xgb"] #["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(models)
    submission.to_csv(f"../output/xgb_baseline.csv", index=False)