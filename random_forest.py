import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Get the arguments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()

# Load the data
df = pd.read_csv(args.trainingdata)
print(df)

# Preprocessing:
# Step 1: Handle missing values (drop rows with missing data)
df.dropna(inplace=True)

# Step 2: Pick out the columns we want to use as inputs (X) and target (Y)
X = df[['x', 'y', 'z']].values
Y = df['gt'].values

# Convert target to string type if it's not already
Y = Y.astype(str)

# Check the class distribution
print(np.unique(Y, return_counts=True))

# Step 3: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print("Train length", len(X_train))
print("Test length", len(X_test))

# Step 4: Train the model using a RandomForestClassifier
model = RandomForestClassifier(n_estimators=300, random_state=42, criterion="gini", max_depth=15, class_weight="balanced").fit(X_train, Y_train)

# Step 5: Evaluate the model's accuracy
testPredictions = model.predict(X_test)
acc = np.average(testPredictions == Y_test)
print("Accuracy", acc)

# Step 6: Compute and plot the confusion matrix
cm = confusion_matrix(Y_test, testPredictions, normalize='false')  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Step 7: More thorough evaluation - ROC Curve and AUC
Y_scores = model.predict_proba(X_test)

# Create a figure for ROC Curve
fig = plt.figure(figsize=(6, 4))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

# Plot the 50% line (diagonal)
plt.plot([0, 1], [0, 1], 'k--')

# Plot ROC curve for the different classes
for idx, className in enumerate(df['gt'].unique()):
    fpr, tpr, thresholds = roc_curve(Y_test == className, Y_scores[:, idx])
    seriesName = "ROC for " + className
    plt.plot(fpr, tpr, label=seriesName)

# Add a legend
plt.legend()

# Compute the AUC (Area Under the Curve)
auc = roc_auc_score(Y_test, Y_scores, multi_class='ovr')
print('AUC', auc)

# Show the plot
plt.show()
