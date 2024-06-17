import joblib
import os
import pandas as pd
import numpy as np
import config
from utils import test_mean_target_encoding

actual_model = "xgb_tuned"

def predict(test, sample, models):
    predictions = None
    label_encoder = joblib.load(os.path.join(config.MODEL_OUTPUT, f"label_encoder.pkl"))

    for model in models:
        for fold in range(config.N_FOLDS):
            clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}_target_enc.pkl"))
            preds = clf.predict_proba(test)
            if fold == 0:
                predictions = preds
            else:
                predictions += preds

        predictions = predictions / config.N_FOLDS

        if model == actual_model:
            ens_preds = predictions
        else:
            ens_preds += predictions

    ens_preds = np.argmax(ens_preds / len(models), axis=1)
    sample.Target = label_encoder.inverse_transform(ens_preds)
    print(sample.head())
    return sample

if __name__ == "__main__":
    train = pd.read_csv(config.TRAINING_FILE)
    test  = pd.read_csv(config.TEST_FILE)
    sample = pd.read_csv(config.SAMPLE_FILE)

    #test = test_mean_target_encoding(train, test, alpha=5)

    models = [actual_model]                #["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(test, sample, models)
    submission.to_csv(f"../output/{actual_model}_target_enc.csv", index=False)