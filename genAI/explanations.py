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

import shap

def explain_prediction(model, sample, feature_names):
    explainer = shap.Explainer(model, feature_names)
    shap_values = explainer(sample)
    shap.plots.waterfall(shap_values[0])