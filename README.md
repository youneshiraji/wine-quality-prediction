# wine-quality-prediction
#wine quality prediction usin randomforest and boosting.

[random_forest_boosting (3).pdf](https://github.com/youneshiraji/wine-quality-prediction/files/14069590/random_forest_boosting.3.pdf)





    import pandas as pd
    data = pd.read_csv('wine.csv',skiprows=35)
    print(data.info())

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB
    None

    X = data.drop('quality', axis=1)
    y = data['quality']

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # On a appliquez le seuil pour classer comme "good" ou "bad", si quality>6 alors c'est good sinon c'est bad.
    seuil_classification = 5
    predictions_classe = ['good' if p > seuil_classification else 'bad' for p in predictions]

    print(predictions_classe)

    ['bad', 'bad', 'bad', 'bad', 'good', 'bad', 'bad', 'bad', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'bad', 'bad', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'good', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'good', 'good', 'bad', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'bad', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'bad', 'bad', 'good', 'good', 'good', 'good', 'bad', 'good', 'good', 'good', 'bad', 'bad', 'good', 'good', 'bad', 'good', 'bad', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'bad', 'good', 'good', 'bad', 'bad', 'good', 'bad', 'good', 'good', 'good', 'bad', 'bad', 'good', 'good', 'good', 'bad', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'good', 'bad', 'bad', 'good', 'bad']

    # On a ajoutez une nouvelle colonne 'quality' au DataFrame
    X_test_with_quality = X_test.copy()
    X_test_with_quality['quality'] = predictions_classe

    print(X_test_with_quality[['quality']])

         quality
    803      bad
    124      bad
    350      bad
    682      bad
    1326    good
    ...      ...
    1259    good
    1295     bad
    1155     bad
    963     good
    704      bad

    [320 rows x 1 columns]

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Précision
    precision = precision_score(y_test, predictions, average='weighted')
    print(f'Précision : {precision * 100}%')

    # Rappel
    recall = recall_score(y_test, predictions, average='weighted')
    print(f'Rappel : {recall * 100}%')

    # F1-Score
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f'F1-Score : {f1 * 100}%')

    Précision : 63.198150012584954%
    Rappel : 65.9375%
    F1-Score : 64.42498546491974%

    /usr/local/lib/python3.10/dist-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))

    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Charger les données
    data = pd.read_csv('wine.csv', skiprows=35)

    # Préparer les données
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer le modèle
    model = GradientBoostingClassifier(random_state=42)

    # Définir les hyperparamètres à optimiser
    param_grid = {
        'n_estimators': [5, 10, 20],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [1, 3, 5]
    }

    # Utiliser la validation croisée pour optimiser les hyperparamètres
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Afficher les meilleurs hyperparamètres
    print("Best parameters: ", grid_search.best_params_)

    # Utiliser le meilleur modèle pour faire des prédictions
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    # Classification
    seuil_classification = 5
    predictions_classe = ['good' if p > seuil_classification else 'bad' for p in predictions]

    Best parameters:  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 20}

    from sklearn.metrics import accuracy_score, recall_score, f1_score, mean_squared_error, r2_score
    import numpy as np

    # Supposons que y_test sont vos vraies étiquettes et predictions sont vos étiquettes prédites
    # y_test = ...
    # predictions = ...

    # Précision
    precision = accuracy_score(y_test, predictions)
    print(f'Précision: {precision}')

    # Rappel
    rappel = recall_score(y_test, predictions, average='macro')
    print(f'Rappel: {rappel}')

    # Score F1
    f1 = f1_score(y_test, predictions, average='macro')
    print(f'Score F1: {f1}')

    # Erreur quadratique moyenne (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f'RMSE: {rmse}')

    # Coefficient de détermination (R²)
    r2 = r2_score(y_test, predictions)
    print(f'R²: {r2}')

    Précision: 0.59375
    Rappel: 0.29036796536796533
    Score F1: 0.29540768497943914
    RMSE: 0.7373940601876312
    R²: 0.16794931185463025
