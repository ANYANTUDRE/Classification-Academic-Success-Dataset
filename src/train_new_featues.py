import os
import argparse
import config
import model_dispatcher
import joblib
import numpy as np
import pandas as pd 
from sklearn import metrics, preprocessing
from utils import feature_engineering
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

cat_features = ['Marital status', 'Application mode', 'Course',
                'Previous qualification', 'Nacionality', "Mother's qualification", 
                "Father's qualification", "Mother's occupation",
                "Father's occupation"]

def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    df = feature_engineering(df, cat_features)
    #print(df.shape)

    # list of numerical columns for feature engineering
    num_features = [c for c in df.columns if c not in cat_features 
                    and c not in ("kfold", "Target")]
    features = [f for f in df.columns if f not in ("kfold", "Target")]

    for col in features:
        # do not encode the numerical columns
        if col not in num_features:
            df.loc[:, col] = df[col].astype(str)   # .fillna("NONE") no need

    # now its time to label encode the features
    for col in features:
        if col not in num_features: 
            # initialize LabelEncoder for each feature column
            lbl = preprocessing.LabelEncoder()
            # fit label encoder on all data
            lbl.fit(df[col])
            df.loc[:, col] = lbl.transform(df[col])

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

    # calculate roc auc
    score = metrics.accuracy_score(y_valid, np.argmax(preds, axis=1))
    print(f"Fold--->{fold}, accuracy={score:.5f}")
    #print(f"Confusion Matrix:\n {metrics.confusion_matrix(y_valid, preds)}")

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.pkl"))

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their types
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)