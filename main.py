from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, recall_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import numpy as np
import time 

def quality(M,XT,YT):

    """ Evaluate the model M """

    Yp = M.predict(XT)
    print('Accuracy: ' + format(accuracy_score(YT,Yp),".3f"))
    print('Balanced Accuracy: ' + format(balanced_accuracy_score(YT,Yp),".3f"))
    print('Confusion Matrix: \n' + str(confusion_matrix(YT,Yp)))

def sanity_check(df):

    """ Checks if the input data frame is valid. """

    if df.isnull().values.any():
        raise ValueError("Input data contains null values.")

    if (df[["C1", "C2", "C3", "C4", "C5"]].nunique(axis=1) < 2).any():
        raise ValueError("Some hands have all cards the same, which is not possible in a standard poker hand.")
    
    if (df[["C1", "C2", "C3", "C4", "C5"]].min(axis=1) < 1).any() or (df[["C1", "C2", "C3", "C4", "C5"]].max(axis=1) > 13).any():
        raise ValueError("Card ranks must be between 1 and 13.")

    if (df[["S1", "S2", "S3", "S4", "S5"]].min(axis=1) < 1).any() or (df[["S1", "S2", "S3", "S4", "S5"]].max(axis=1) > 4).any():
        raise ValueError("Card suits must be between 1 and 4.")

def extract_features_poker_hand(df):

    """ Extracts features from the raw data frame of cards. """

    try: 
        sanity_check(df) 
    except ValueError as e:
        print(e)
        exit(1)

    features = pd.DataFrame()

    # 0. Extract card number (rank) and sort them then put them in new columns C1, C2, C3, C4, C5
    ranks = df[["C1", "C2", "C3", "C4", "C5"]].values.copy()
    ranks.sort(axis=1)
    for i in range(5):
        features[f"C{i+1}"] = ranks[:, i]

    # 1. Calculate the first and the second element that appears most frequently 
    def counts_ranks(row):
        """
        Returns the counts of each unique value in the row, sorted in descending order.
        """
        unique, counts = np.unique(row, return_counts=True)
        return sorted(counts, reverse=True)

    counts = np.array([counts_ranks(r) for r in ranks], dtype=object)
    features["Max_Freq"] = [c[0] for c in counts]
    features["Second_Max_Freq"] = [c[1] for c in counts]

    # 2. Calculate if all the suits are the same (Flush)
    features["Is_Flush"] = (df[["S1", "S2", "S3", "S4", "S5"]].nunique(axis=1) == 1).astype(int)

    # 3. Calculate if the hand is a Straight (standard or royal)
    is_straight = (ranks[:, 4] - ranks[:, 0] == 4) & (features["Max_Freq"] == 1)
    is_royal_straight = (
        (ranks[:, 0] == 1) & (ranks[:, 1] == 10) & (ranks[:, 2] == 11) & 
        (ranks[:, 3] == 12) & (ranks[:, 4] == 13)
    )
    features["Is_Straight"] = (is_straight | is_royal_straight).astype(int)

    return features

# ---------- Set up of the data ----------

print("Preprocessing data...")
start = time.time()

columns = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5","CLASS"]
path = "poker_hand/poker-hand-"

train = pd.read_csv(path + "training-true.data", header=None, names=columns)
test = pd.read_csv(path + "testing.data", header=None, names=columns)

X_train = extract_features_poker_hand(train)
Y_train = train["CLASS"]

X_test = extract_features_poker_hand(test)
Y_test = test["CLASS"]

end = time.time()
print(f"Data preprocessing completed in {end - start:.2f} seconds.")
print("-----------------------------------------------------------")

# ---------- Set up of the Model ----------

print("Setting up the model...")
start = time.time()

grid = {
    'max_features': [3, 5, 7]
}

M = GridSearchCV(
    estimator=RandomForestClassifier(class_weight='balanced'),
    param_grid=grid, 
    cv=3,
    n_jobs=-1
)

end = time.time()
print(f"Model set up completed in {end - start:.2f} seconds.")
print("-----------------------------------------------------------")

# ---------- Training ----------

print("Training with balanced weights...")
start = time.time()

M.fit(X_train, Y_train)

end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")
print("-----------------------------------------------------------")

# ---------- Testing ----------

print("Testing...")
start = time.time()

best = M.best_estimator_
# print('max_features best: ' + str(M.best_params_['max_features']))
quality(best, X_test, Y_test)

end = time.time()
print(f"Testing completed in {end - start:.2f} seconds.")
print("-----------------------------------------------------------")

# ---------- Single hand ----------

def predict_custom_hand(model, hand_list):
    columns = ["S1","C1","S2","C2","S3","C3","S4","C4","S5","C5"]
    df_single = pd.DataFrame([hand_list], columns=columns)

    features_single = extract_features_poker_hand(df_single)

    if(features_single is None):
        print("Error in feature extraction. Please check the input data.")
        exit(1)

    prediction = model.predict(features_single)[0]

    hand_names = {
        0: "Nothing", 1: "One Pair", 2: "Two Pairs", 3: "Three of a Kind",
        4: "Straight", 5: "Flush", 6: "Full House", 7: "Four of a Kind",
        8: "Straight Flush", 9: "Royal Flush"
    }

    print(f"Hand: {hand_list}")
    print(f"Predicted Class: {prediction} ({hand_names.get(prediction)})")
    return prediction

# my_hand = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# predict_custom_hand(best, my_hand)
