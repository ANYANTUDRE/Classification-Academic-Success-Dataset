from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgbm
import catboost

models = { 
    #"extra": ensemble.ExtraTreesClassifier(), no!!!

    "xgb": xgb.XGBClassifier(n_jobs=-1, 
                             #eta=0.01, 
                             #gamma=0.5, 
                             #max_depth=7
                             ),            # plutot pas mal

    "xgb_tuned1": xgb.XGBClassifier(**{'grow_policy': 'depthwise', 'tree_method': 'hist', 'enable_categorical': True, 
                                      'gamma': 0, 'n_estimators': 768, 'learning_rate': 0.026111403303690425, 
                                      'max_depth': 8, 'reg_lambda': 26.648168065161098, 
                                      'min_child_weight': 1.0626186255116183, 'subsample': 0.8580490989206254, 'colsample_bytree': 0.5125814118774029
                                      }),
    
    "xgb_tuned2": xgb.XGBClassifier(**{'grow_policy': 'depthwise', 'learning_rate': 0.06150883051411711, 
                                       'n_estimators': 822, 'max_depth': 5, 'reg_lambda': 12.695544291635334, 
                                       'min_child_weight': 25.289808052903343, 'subsample': 0.9831949009517902, 
                                       'colsample_bytree': 0.2687930272243655, 'tree_method': 'hist', 'enable_categorical': True, 'gamma': 0
                                       }),
                                       
    "xgb_tuned3": xgb.XGBClassifier(**{'grow_policy': 'depthwise', 'learning_rate': 0.04104089631389812, 
                                      'n_estimators': 1311, 'max_depth': 5, 'reg_lambda': 29.548955808402486, 
                                      'min_child_weight': 17.58377776073493, 'subsample': 0.9141573846486278, 
                                      'colsample_bytree': 0.4000772723424121, 'tree_method': 'hist', 
                                      'enable_categorical': True, 'gamma': 0
                                      }),

    "lgbm": lgbm.LGBMClassifier(n_jobs=-1),
    'gbm': ensemble.GradientBoostingClassifier(),
    'cat': catboost.CatBoostClassifier(verbose=False),
}