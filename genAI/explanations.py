def create_description(row):
    return (
        f"Passager de classe {row['Pclass']}, "
        f"{'homme' if row['Sex']=='male' else 'femme'}, "
        f"âgé(e) de {row['Age']} ans, "
        f"ayant payé {row['Fare']}€, "
        f"embarqué(e) à {row['Embarked']}. "
        f"{'A survécu' if row['Survived'] == 1 else 'N’a pas survécu'}."
    )

def explain_prediction_text(row, prediction):
    return (
        f"Ce passager est un(e) {'homme' if row['Sex']=='male' else 'femme'} "
        f"de {row['Age']} ans, classe {row['Pclass']}, "
        f"ayant payé {row['Fare']}€, embarqué(e) à {row['Embarked']}. "
        f"Le modèle prédit qu’il/elle "
        f"{'a survécu' if prediction == 1 else 'n’a pas survécu'} "
        f"car les passagers similaires dans l'entraînement avaient le même profil."
    )

import joblib

import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain_prediction(model, sample, background_data=None, feature_names=None):
    """
    Affiche une explication SHAP pour un individu donné avec matplotlib.
    - model: modèle entraîné (scikit-learn)
    - sample: DataFrame ou Series (1 ligne) représentant l'individu à expliquer
    - background_data: DataFrame (exemple : X_train), utilisé pour créer l'explainer
    - feature_names: liste des noms de features, optionnel
    """

    # Gestion format sample
    if isinstance(sample, pd.Series):
        sample = pd.DataFrame([sample])
    elif not isinstance(sample, pd.DataFrame):
        if feature_names is not None:
            sample = pd.DataFrame([sample], columns=feature_names)
        else:
            raise ValueError("sample doit être DataFrame ou Series, ou fournir feature_names")

    # Gestion background_data
    if background_data is None:
        raise ValueError("Il faut fournir background_data (ex : X_train) pour créer l'explainer SHAP")

    # Créer explainer SHAP avec dataset de fond
    explainer = shap.Explainer(model, background_data)
    
    # Calculer valeurs SHAP pour le sample
    shap_values = explainer(sample)

    # Afficher avec matplotlib
    plt.figure(figsize=(10,6))
    shap.plots.waterfall(shap_values[0], show=False)  # show=False pour contrôler l'affichage

    plt.tight_layout()
    plt.show()


def create_description_regression(row, prediction, target_name="valeur prédite"):
    return (
        f"Passager de classe {row['Pclass']}, "
        f"{'homme' if row['Sex'] == 'male' else 'femme'}, "
        f"âgé(e) de {row['Age']} ans, "
        f"ayant payé {row['Fare']}€, "
        f"embarqué(e) à {row['Embarked']}. "
        f"Prédiction du modèle ({target_name}) : {prediction:.2f}."
    )




