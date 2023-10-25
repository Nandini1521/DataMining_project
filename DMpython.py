import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def preprocess_data(data, normalization_method='zscore', num_imputation_method='mean', cat_imputation_method='most_frequent'):
    # Separate numerical and categorical features
    numerical_features = data.iloc[:, :100]
    categorical_features = data.iloc[:, 100:128]
    labels = data.iloc[:, -1]

    # Step 1: Impute missing values in numerical features
    if num_imputation_method == 'mean':
        imputer = SimpleImputer(strategy="mean")
        
    elif num_imputation_method == 'class_mean':
        # Impute missing numerical values by the average of the feature's class-specific values
        numerical_features_imputed = data.groupby(labels).transform(lambda x: x.fillna(x.mean()))
        return numerical_features_imputed

    numerical_features_imputed = imputer.fit_transform(numerical_features)

    # Step 2: Normalize numerical features
    if normalization_method == 'zscore':
        scaler = StandardScaler()
    elif normalization_method == 'minmax':
        scaler = MinMaxScaler()

    numerical_features_normalized = scaler.fit_transform(numerical_features_imputed)

    # Step 4: Impute missing values in categorical features
    if cat_imputation_method == 'most_frequent':
        categorical_features_imputed = categorical_features.fillna(categorical_features.mode().iloc[0])
    elif cat_imputation_method == 'class_most_frequent':
        categorical_features_imputed = categorical_features.groupby(labels).transform(lambda x: x.fillna(x.mode().iloc[0]))
        
    # Combine the preprocessed numerical and categorical features
    preprocessed_data = pd.concat([pd.DataFrame(numerical_features_normalized), categorical_features_imputed, labels], axis=1)


    # Reset the index
    preprocessed_data = preprocessed_data.reset_index(drop=True)

    return preprocessed_data


def classifiers(dataset):
     # Separate features and labels
    X = dataset.iloc[:, :-1]  # Features
    X.columns = X.columns.astype(str)
    y = dataset.iloc[:, -1]   # Labels

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter ranges
    decisionClass= {'max_depth': [5, 15, 25, 30], 'min_samples_split': [5, 10, 15]}
    randomClass= {'n_estimators': [50, 200, 500], 'max_depth': [5, 15, 25, 30]}
    kClass= {'n_neighbors': [1, 2, 3, 5, 7, 10], 'p': [1, 2, 3] }
    gaussClass = {'var_smoothing': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    
    
    best_classifiers = {}
    
    grid_search1 = GridSearchCV(DecisionTreeClassifier(), decisionClass, scoring='f1_macro', cv=5, n_jobs=-1)
    grid_search1.fit(X_train, y_train)
    best_clf1 = grid_search1.best_estimator_
    best_classifiers['DecisionTreeClassifier'] = best_clf1
    
    grid_search2 = GridSearchCV(RandomForestClassifier(), randomClass, scoring='f1_macro', cv=5, n_jobs=-1)
    grid_search2.fit(X_train, y_train)
    best_clf2 = grid_search2.best_estimator_
    best_classifiers['RandomForestClassifier'] = best_clf2
    
    grid_search3 = GridSearchCV(KNeighborsClassifier(), kClass, scoring='f1_macro', cv=5, n_jobs=-1)
    grid_search3.fit(X_train, y_train)
    best_clf3 = grid_search3.best_estimator_
    best_classifiers['KNeighborsClassifier'] = best_clf3
    
    grid_search4 = GridSearchCV(GaussianNB(), gaussClass, scoring='f1_macro', cv=5, n_jobs=-1)
    grid_search4.fit(X_train, y_train)
    best_clf4 = grid_search4.best_estimator_
    best_classifiers['GaussianNB'] = best_clf4

    f1scores= []
    # Model evaluation
    for clf_name, clf in best_classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        f1scores.append(f1_score(y_val, y_pred, average='macro'))

    return best_classifiers, f1scores

def preprocess_combos(data):
    # Contains pre-processing combinations:
    preprocess_dic = {
        1: ['mean', 'zscore', 'most_frequent'],
        2: ['mean', 'zscore', 'class_most_frequent'],
        3: ['mean', 'minmax', 'most_frequent'],
        4: ['mean', 'minmax', 'class_most_frequent'],
        5: ['class_mean', 'zscore', 'most_frequent'],
        6: ['class_mean', 'zscore', 'class_most_frequent'],
        7: ['class_mean', 'minmax', 'most_frequent'],
        8: ['class_mean', 'minmax', 'class_most_frequent']
    }

    preprocessed_data = []
    for key in preprocess_dic:
        combo = preprocess_dic[key]
        num_imputation = combo[0]
        normalization = combo[1]
        cat_imputation = combo[2]

        preprocessed_data.append(preprocess_data(data, normalization_method=normalization, num_imputation_method=num_imputation, cat_imputation_method=cat_imputation))

    return preprocessed_data

def main():
    main_train = pd.read_csv("dm_train.csv")
    add_train = pd.read_csv("add_train.csv")
    complete_train = pd.concat([main_train, add_train])
    complete_train = complete_train.reset_index(drop=True)
    
    all_traindatasets = preprocess_combos(complete_train)

    best_class_for_each_dataset = {}
    f1_score_for_each_dataset = {}
    
    for count, dataset in enumerate(all_traindatasets, 1):
        bestclass, f1score = classifiers(dataset)
        best_class_for_each_dataset[count] = bestclass
        f1_score_for_each_dataset[count] = f1score

    flat_data = [(key, value, idx) for key, values in f1_score_for_each_dataset.items() for idx, value in enumerate(values)]
    sorted_data = sorted(flat_data, key=lambda x: x[1], reverse=True)
    top_three = sorted_data[:3]
    
    top_3_keys = [item[0] for item in top_three]
    top_3_model_index = [item[2] for item in top_three]
    
    combos_corresponding_to_keys = [best_class_for_each_dataset[key] for key in top_3_keys]

    top3_combos = []
    for dic in combos_corresponding_to_keys:
        values_list = list(dic.values())
        top3_combos.append(values_list)
    
    best_classifiers = []
    for i in range(3):
        index = top_3_model_index[i]
        best_classifiers.append(top3_combos[i][index])

    ensemble_classifier = VotingClassifier(estimators=[
        ('clf1', best_classifiers[0]), 
        ('clf2', best_classifiers[1]), 
        ('clf3', best_classifiers[2])
    ], voting='hard')
    
    main_test = pd.read_csv("dm_test.csv")
    all_testdatasets = preprocess_combos(main_test)
    full_test_data = []

    for test_data in all_testdatasets:
        full_data = test_data.iloc[:, :128]
        full_test_data.append(full_data)

    y_pred_list = []
    f1_score_list = []
    accuracy_list = []

    for index in range(8):
        X = all_traindatasets[index].iloc[:, :-1]
        X.columns = X.columns.astype(str)
        y = all_traindatasets[index].iloc[:, -1]

        full_test_data[index].columns = full_test_data[index].columns.astype(str)

        num_folds = 5

        cv_f1_scores = cross_val_score(ensemble_classifier, X, y, cv=num_folds, scoring=make_scorer(f1_score, average='macro'))
        cv_accuracy_scores = cross_val_score(ensemble_classifier, X, y, cv=num_folds, scoring=make_scorer(accuracy_score))

        ensemble_classifier.fit(X, y)
        y_pred = ensemble_classifier.predict(full_test_data[index])
        y_pred_list.append(y_pred)

        #print(f"Cross-Validation F1 Scores for Dataset {index + 1}: {cv_f1_scores}")
        print(f"Mean F1 Score: {cv_f1_scores.mean()}")
        f1_score_list.append(cv_f1_scores.mean())

        #print(f"Cross-Validation Accuracy Scores for Dataset {index + 1}: {cv_accuracy_scores}")
        print(f"Mean Accuracy: {cv_accuracy_scores.mean()}")
        accuracy_list.append(cv_accuracy_scores.mean())

    f1mean = sum(f1_score_list) / len(f1_score_list)
    print("F1 Mean:", f1mean)

    accuracymean = sum(accuracy_list) / len(accuracy_list)
    print("Accuracy:", accuracymean)

    def majority_vote(row):
        unique, counts = np.unique(row, return_counts=True)
        return unique[np.argmax(counts)]

    majority_vote_result = np.apply_along_axis(majority_vote, axis=0, arr=y_pred_list)
    
    df = pd.DataFrame(majority_vote_result, columns=['Prediction'])
    df['Prediction'] = df['Prediction'].astype(int)
    
    csv_file_path = 's4593940.csv'
    df.to_csv(csv_file_path, index=False, header=False)


if __name__ == "__main__":
    main()
