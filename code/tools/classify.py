from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


def classify_oversampled(data, connections=False):
    triuim1 = np.triu_indices_from(st_matrices[0], k=1)
    X = [np.array(matrix)[triuim1] for matrix in data]

    # Convert the list of upper triangles to a 2D array
    X = np.array(X)
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

    if connections:

        # Apply SMOTE to the data
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Shuffle the oversampled data
        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=24)

        # Train the final model with the best hyperparameters
        final_clf = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
        final_clf.fit(X_train, y_train)

        # Predictions on the test set
        preds = final_clf.predict(X_test)

        # Evaluate performance
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        cnf_matrix = confusion_matrix(y_test, preds)

        n_sv = final_clf.support_vectors_.shape[0]
        feature_importances = np.abs(np.dot(final_clf.dual_coef_, final_clf.support_vectors_)).flatten() / n_sv

        top_indices = np.argsort(feature_importances)[-7:][::-1]

        # Get the row and column indices for the flattened indices
        row_indices, col_indices = np.triu_indices(76, k=1)

        # Map the flattened indices to the original matrix indices
        top_original_indices = list(zip(row_indices[top_indices], col_indices[top_indices]))

        # Print the result
        for idx, original_idx in enumerate(top_original_indices):
            # print(f"Top {idx + 1} Feature: Importance {feature_importances[top_indices[idx]]:.4f}, Original Matrix Index: {original_idx+}")
            print(
                f"Top {idx + 1} Feature: Importance {feature_importances[top_indices[idx]]:.4f}, Original Matrix Index: {original_idx[0] + 1, original_idx[1] + 1}")


def classify_oversampled_skf2(data, connections=False, title=None):
    triuim1 = np.triu_indices_from(st_matrices[0], k=1)
    X = [np.array(matrix)[triuim1] for matrix in data]

    # we convert the list of upper triangles to a 2D array
    X = np.array(X)
    y = np.array(labels)

    # we use SMOTE for oversampling
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # StratifiedKFold with n_splits=5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_params = None

    # SVM model
    clf = SVC()

    # Grid search
    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 200], "gamma": [0.001, 0.01, 0.1, 1, 10]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=11)
    grid_search.fit(X, y)

    # we use the best hyperparameters for k-fold cross-validation
    best_params = grid_search.best_params_

    # variables to store the best results
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_confusion_matrix = None

    progress_bar = tqdm(total=skf.get_n_splits(X, y), desc="Folds", position=0, leave=True)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # we apply SMOTE to the training data
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

        # we train the model with the best parameters
        best_clf = SVC(C=best_params['C'], gamma=best_params['gamma'])
        best_clf.fit(X_resampled, y_resampled)

        # Predictions
        preds = best_clf.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cnf_matrix = confusion_matrix(y_test, preds)

        if accuracy > best_accuracy:
            best_accuracy = round(accuracy, 2)
            best_precision = round(precision, 2)
            best_recall = round(recall, 2)
            best_f1 = round(f1, 2)
            best_confusion_matrix = cnf_matrix

        progress_bar.update(1)

    progress_bar.close()

    # Printing the best results
    print("Best results {}:".format(title))
    print("Best Accuracy: {:.4f}".format(best_accuracy))
    print("Best Precision: {:.4f}".format(best_precision))
    print("Best Recall: {:.4f}".format(best_recall))
    print("Best F1 Score: {:.4f}".format(best_f1))
    print("Best Confusion Matrix:")
    print(best_confusion_matrix)
    print("Best Parameters (C and Gamma):", best_params)

    if connections:
        # we apply SMOTE to the data
        X_resampled, y_resampled = smote.fit_resample(X, y)

        X_resampled, y_resampled = shuffle(X_resampled, y_resampled, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=24)

        # we train the final model with the best hyperparameters
        final_clf = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
        final_clf.fit(X_train, y_train)

        # Predictions
        preds = final_clf.predict(X_test)

        # Evaluation
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        cnf_matrix = confusion_matrix(y_test, preds)

        n_sv = final_clf.support_vectors_.shape[0]
        feature_importances = np.abs(np.dot(final_clf.dual_coef_, final_clf.support_vectors_)).flatten() / n_sv

        top_indices = np.argsort(feature_importances)[-7:][::-1]

        # row and column indices for the flattened indices
        row_indices, col_indices = np.triu_indices(76, k=1)

        # we map the flattened indices to the original matrix indices
        top_original_indices = list(zip(row_indices[top_indices], col_indices[top_indices]))

        # Printing the result
        for idx, original_idx in enumerate(top_original_indices):
            print(
                f"Node connection: {original_idx[0] + 1, original_idx[1] + 1}, Importance: {feature_importances[top_indices[idx]]:.4f}")

    return best_accuracy, best_precision, best_recall, best_f1