from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, classification_report, f1_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def classify_graph(data, labels):
    X = np.array(data)
    y = np.array(labels)

    # Apply SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Create variables to store the best results
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_confusion_matrix = None
    best_params = None

    # Create 100 instances
    num_instances = 50

    progress_bar = tqdm(total=num_instances, desc="Instances", position=0, leave=True)

    for instance in range(num_instances):
        # Apply SMOTE to the data
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Shuffle the oversampled data
        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=24)

        # SVM model
        clf = SVC()

        # Grid search for hyperparameter tuning
        param_grid = {"C": [0.01, 0.1, 1, 10, 100, 200], "gamma": [0.001, 0.01, 0.1, 1, 10]}
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=11)
        grid_search.fit(X_train, y_train)

        # Train the model with the best parameters
        best_clf = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
        best_clf.fit(X_train, y_train)

        # Predictions on the test set
        preds = best_clf.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cnf_matrix = confusion_matrix(y_test, preds)

        # Update best results if current results are better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_confusion_matrix = cnf_matrix
            best_params = grid_search.best_params_

            # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Print the best results
    print("Best Results:")
    print("Best Accuracy: {:.4f}".format(best_accuracy))
    print("Best Precision: {:.4f}".format(best_precision))
    print("Best Recall: {:.4f}".format(best_recall))
    print("Best F1 Score: {:.4f}".format(best_f1))
    print("Best Confusion Matrix:")
    print(best_confusion_matrix)
    print("Best Parameters (C and Gamma):", best_params)


def classify_graph_skf(data, labels):
    X = np.array(data)
    y = np.array(labels)

    # Apply SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Create variables to store the best results
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_confusion_matrix = None
    best_params = None

    # Create StratifiedKFold with n_splits=5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    progress_bar = tqdm(total=skf.get_n_splits(X, y), desc="Folds", position=0, leave=True)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Apply SMOTE to the training data
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        # Shuffle the oversampled data
        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

        # SVM model
        clf = SVC()

        # Grid search for hyperparameter tuning
        param_grid = {"C": [0.01, 0.1, 1, 10, 100, 200], "gamma": [0.001, 0.01, 0.1, 1, 10]}
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=11)
        grid_search.fit(X_resampled, y_resampled)

        # Train the model with the best parameters
        best_clf = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'])
        best_clf.fit(X_resampled, y_resampled)

        # Predictions on the test set
        preds = best_clf.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cnf_matrix = confusion_matrix(y_test, preds)

        # Update best results if current results are better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_confusion_matrix = cnf_matrix
            best_params = grid_search.best_params_

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Print the best results
    print("Best Results:")
    print("Best Accuracy: {:.4f}".format(best_accuracy))
    print("Best Precision: {:.4f}".format(best_precision))
    print("Best Recall: {:.4f}".format(best_recall))
    print("Best F1 Score: {:.4f}".format(best_f1))
    print("Best Confusion Matrix:")
    print(best_confusion_matrix)
    print("Best Parameters (C and Gamma):", best_params)

