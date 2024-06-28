import config
import model_dispatcher
import joblib
import numpy as np
import pandas as pd 
from sklearn import metrics, preprocessing
import os
from utils import mean_target_encoding, feature_engineering


def run(df, fold, model):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop(["Target", "kfold"], axis=1).values
    y_train = df_train.Target.values

    x_valid = df_valid.drop(["Target", "kfold"], axis=1).values
    y_valid = df_valid.Target.values

    # initialize simple decision tree classifier from sklearn
    clf = model_dispatcher.models[model]
    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict_proba(x_valid)

    # calculate accuracy
    score = metrics.accuracy_score(y_valid, np.argmax(preds, axis=1))
    print(f"Fold--->{fold}, accuracy={score:.5f}")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}_target_enc.pkl"))
    return score


if __name__ == "__main__":
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    df = mean_target_encoding(df, alpha=2)
    #df = feature_engineering(df)
    print(df.head(1))

    scores = []
    for fold in range(config.N_FOLDS):
        score = run(df=df, fold=fold, model="xgb_tuned1")
        scores.append(score)

    print(f"Mean score: {np.mean(scores)}")
    print(f"Overall score: {np.mean(scores) - np.std(scores)}")