import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

# Load data from a CSV using pandas and return it.
# The expected format is three columns for inputs ('x', 'y', 'z') and one column for the target ('gt').
def loadData(file):
    # Load the CSV file
    df = pd.read_csv(file)
    # Drop missing values
    df.dropna(inplace=True)
    # Pick out the columns we want to use as inputs
    X = df[['x', 'y', 'z']].values
    # Pick out the column we want to use as the output
    Y = df['gt'].values.astype(str)
    return X, Y

# Split the data into train and test sets
def splitData(X, Y, test_size=0.2):
    # Use the scikit-learn function
    return train_test_split(X, Y, test_size=test_size, random_state=42)

# Build and train the model using a RandomForestClassifier
def buildModel(X, Y):
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        criterion="gini",
        max_depth=15,
        class_weight="balanced"
    )
    model.fit(X, Y)
    return model

# Assess the model's performance and return accuracy
def assessModel(model, X, Y):
    # Get predictions and compute accuracy
    predictions = model.predict(X)
    acc = np.average(predictions == Y)
    return acc, predictions

# Train the model, evaluate it, and (optionally) save it
def trainModel(dataFile, modelSavePath=None):
    # Load and preprocess the data
    X, Y = loadData(dataFile)
    # Split the data
    X_train, X_test, Y_train, Y_test = splitData(X, Y)
    print("Train length:", len(X_train))
    print("Test length:", len(X_test))
    
    # Build and train the model
    model = buildModel(X_train, Y_train)

    # Evaluate the model on the test data
    acc, predictions = assessModel(model, X_test, Y_test)
    print("Test Accuracy:", acc)

    # Compute and display the confusion matrix
    cm = confusion_matrix(Y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute and plot the ROC curve
    Y_scores = model.predict_proba(X_test)
    plotROCCurve(Y_test, Y_scores, np.unique(Y))
    
    # Compute AUC
    auc = roc_auc_score(Y_test, Y_scores, multi_class='ovr')
    print("AUC:", auc)

    # Save the model if a save path is provided
    if modelSavePath:
        print("Saving model to", modelSavePath)
        mlflow.sklearn.save_model(model, modelSavePath)
    
    return model

# Plot ROC curves for each class
def plotROCCurve(Y_test, Y_scores, classes):
    plt.figure(figsize=(6, 4))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal (50% line)

    for idx, className in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y_test == className, Y_scores[:, idx])
        plt.plot(fpr, tpr, label=f"ROC for {className}")
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to the dataset file')
    parser.add_argument('--model', default=None, help='Path to save the trained model')
    args = parser.parse_args()
    # Enable MLflow autologging
    mlflow.autolog()
    trainModel(args.data, args.model)
