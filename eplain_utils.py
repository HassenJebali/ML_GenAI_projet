
import shap
import pandas as pd

def explain_prediction(model, sample, feature_names=None):
    """
    Affiche une explication SHAP pour un individu donné.
    - model: modèle entraîné (scikit-learn)
    - sample: DataFrame (1 ligne) représentant l'individu à expliquer
    - feature_names: liste des noms de features, optionnel si sample a déjà les bons noms de colonnes
    """
    # S'assurer que sample est un DataFrame
    if not isinstance(sample, pd.DataFrame):
        if feature_names is not None:
            sample = pd.DataFrame([sample], columns=feature_names)
        else:
            raise ValueError("sample doit être un DataFrame ou feature_names doit être fourni")
    
    # Sélectionner le bon explainer
    explainer = shap.Explainer(model, sample)
    shap_values = explainer(sample)
    shap.plots.waterfall(shap_values[0])
    