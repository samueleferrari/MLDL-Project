from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, recall_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import numpy as np

def quality(M,XT,YT):

    """ Evaluate the model M """

    Yp = M.predict(XT)
    print('Accuracy: ' + format(accuracy_score(YT,Yp),".3f"))
    print('Balanced Accuracy: ' + format(balanced_accuracy_score(YT,Yp),".3f"))
    print('Confusion Matrix: \n' + str(confusion_matrix(YT,Yp)))

def counts_ranks(row):
    """
    Returns the counts of each unique value in the row, sorted in descending order.
    """
    unique, counts = np.unique(row, return_counts=True)

    if(len(counts) == 1):
        raise ValueError("In this hand all the cards are the same, which is not possible in a standard poker hand.")

    return sorted(counts, reverse=True)

def extract_features_poker_hand(df):

    """ Extracts features from the raw data frame of cards. """

    features = pd.DataFrame()

    # Extract card number (rank) and sort them 
    ranks = df[["C1", "C2", "C3", "C4", "C5"]].values.copy()
    ranks.sort(axis=1)

    # Insert the rank in the new features dataframe
    for i in range(5):
        features[f"R{i+1}"] = ranks[:, i]

    # Calculate the first and the second element that appears most frequently 
    try:

        counts = np.array([counts_ranks(r) for r in ranks], dtype=object)
        features["Max_Freq"] = [c[0] for c in counts]
        features["Second_Max_Freq"] = [c[1] for c in counts]

    except ValueError as e:
        print(e)
        return None

    # Calculate if all the suits are the same (Flush)
    features["Is_Flush"] = (df[["S1", "S2", "S3", "S4", "S5"]].nunique(axis=1) == 1).astype(int)

    # Calculate if the hand is a Straight (either standard or Ace-high)
    is_straight = (ranks[:, 4] - ranks[:, 0] == 4) & (features["Max_Freq"] == 1)
    is_royal_straight = (
        (ranks[:, 0] == 1) & (ranks[:, 1] == 10) & (ranks[:, 2] == 11) & 
        (ranks[:, 3] == 12) & (ranks[:, 4] == 13)
    )
    features["Is_Straight"] = (is_straight | is_royal_straight).astype(int)

    return features

def predict_custom_hand(model, hand_list):
    columns = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5"]
    df_single = pd.DataFrame([hand_list], columns=columns)

    features_single = extract_features_poker_hand(df_single)

    if(features_single is None):
        print("Error in feature extraction. Please check the input data.")
        exit(1)

    prediction = model.predict(features_single)[0]

    hand_names = {
        0: "Niente", 1: "Coppia", 2: "Doppia Coppia", 3: "Tris",
        4: "Scala", 5: "Colore", 6: "Tris e Doppia Coppia", 7: "Poker",
        8: "Scala Colore", 9: "Scala Reale"
    }

    # hand_names = {
    #     0: "Nothing", 1: "One Pair", 2: "Two Pairs", 3: "Three of a Kind",
    #     4: "Straight", 5: "Flush", 6: "Full House", 7: "Four of a Kind",
    #     8: "Straight Flush", 9: "Royal Flush"
    # }

    print(f"Hand: {hand_list}")
    print(f"Predicted Class: {prediction} ({hand_names.get(prediction)})")
    return prediction

# ---------- Set up of the data ----------

columns = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5","CLASS"]
path = "poker_hand/poker-hand-"

train = pd.read_csv(path + "training-true.data", header=None, names=columns)
test = pd.read_csv(path + "testing.data", header=None, names=columns)

X_train = extract_features_poker_hand(train)
Y_train = train["CLASS"]


X_test = extract_features_poker_hand(test)
Y_test = test["CLASS"]

if(X_train is None or X_test is None):
    print("Error in feature extraction. Please check the input data.")
    exit(1)

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

# ---------- Testing ----------

best = M.best_estimator_
print('max_features best: ' + str(M.best_params_['max_features']))
quality(best, X_test, Y_test)

# ---------- Single hand ----------
my_hand = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
predict_custom_hand(best, my_hand)
