import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import joblib
import os

if __name__ == "__main__":
    # import the dataset
    df = pd.read_csv(config.BASE_TRAINING_FILE)

    # create new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch & encode target
    y = df.Target.values
    label_encoder = LabelEncoder()
    df.Target = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(config.MODEL_OUTPUT, f"{label_encoder}.pkl"))

    # initiate the Stratified Kfold class from model_selection module
    skf = StratifiedKFold(n_splits=config.N_FOLDS)
    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(skf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] = fold

    # save the new csv with kfold column
    df.to_csv("../input/train_folds.csv", index=False)
    print("Folds created successfully !!!")
