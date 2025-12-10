import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

modelo.fit(X_train_scaled, y_train)

joblib.dump(modelo, "modelo_fraude.pkl")
joblib.dump(scaler, "scaler.pkl")

fraudes_reais = df[df["Class"] == 1].iloc[:, 1:29].values
np.save("pca_fraudes_reais.npy", fraudes_reais)

print("Treinamento finalizado.")
print("Arquivos salvos: modelo_fraude.pkl, scaler.pkl, pca_fraudes_reais.npy")
