from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import numpy as np

def quality(M,XT,YT):

    """ Evaluate the model M """

    Yp = M.predict(XT)
    print('Accuracy: ' + format(accuracy_score(YT,Yp),".3f"))
    print('Balanced Accuracy: ' + format(balanced_accuracy_score(YT,Yp),".3f"))
    print('Confusion Matrix: \n' + str(confusion_matrix(YT,Yp)))

def extract_features_poker_hand(df):

    """ Extracts features from the raw data frame of cards. """

    ranks = df[["C1", "C2", "C3", "C4", "C5"]].values
    ranks.sort(axis=1)
    
    features = pd.DataFrame()
    
    for i in range(5):
        features[f"R{i+1}"] = ranks[:, i]
    
    def get_counts(row):
        unique, counts = np.unique(row, return_counts=True)
        return sorted(counts, reverse=True)

    counts = np.array([get_counts(r) for r in ranks], dtype=object)
    
    features["Max_Freq"] = [c[0] for c in counts]
    features["Second_Freq"] = [c[1] if len(c) > 1 else 0 for c in counts]
    features["Is_Flush"] = (df[["S1", "S2", "S3", "S4", "S5"]].nunique(axis=1) == 1).astype(int)

    is_std_straight = (ranks[:, 4] - ranks[:, 0] == 4) & (features["Max_Freq"] == 1)
    is_ace_high_straight = (
        (ranks[:, 0] == 1) & (ranks[:, 1] == 10) & (ranks[:, 2] == 11) & 
        (ranks[:, 3] == 12) & (ranks[:, 4] == 13)
    )

    features["Is_Straight"] = (is_std_straight | is_ace_high_straight).astype(int)
    features["Is_Straight_Flush"] = features["Is_Flush"] & features["Is_Straight"]
    
    return features

# ---------- Set up of the data ----------

columns = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5","CLASS"]
path = "poker_hand/poker-hand-"

train = pd.read_csv(path + "training-true.data", header=None, names=columns)
test = pd.read_csv(path + "testing.data", header=None, names=columns)

X_train = extract_features_poker_hand(train)
Y_train = train["CLASS"]

X_test = extract_features_poker_hand(test)
Y_test = test["CLASS"]

# ---------- Set up of the Model ----------

# Used to handle class imbalance 
weights = compute_sample_weight(class_weight='balanced', y=Y_train)

# M = HistGradientBoostingClassifier(
#     random_state=42, 
#     max_iter=1000,
#     max_leaf_nodes=127,
#     learning_rate=0.05,
#     early_stopping=True,
#     n_iter_no_change=20
# )

# M = RandomForestClassifier(
#     random_state=42,
#     max_features=5,
#     class_weight="balanced_subsample",
#     n_estimators=1000,
#     n_jobs=-1
# )

grid = {
    'max_features': [2, 3, 5, 7, 10]
}

M = GridSearchCV(
    estimator=RandomForestClassifier(), 
    param_grid=grid, 
    cv=3,
    n_jobs=-1
)

# ---------- Training ----------

print("Training with balanced weights...")
M.fit(X_train, Y_train, sample_weight=weights)

# ---------- Evaluation ----------

best = M.best_estimator_
print('max_features best: ' + str(M.best_params_['max_features']))
quality(best, X_test, Y_test)
