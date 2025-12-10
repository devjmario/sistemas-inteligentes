import numpy as np
import joblib
import random

modelo = joblib.load("modelo_fraude.pkl")
scaler = joblib.load("scaler.pkl")
fraudes_reais = np.load("pca_fraudes_reais.npy")

"""
def gerar_transacao(fraudulenta=False):
    return {
        "fraudulenta": fraudulenta,
        "valor": round(random.uniform(1, 3000), 2),
        "distancia_km": random.uniform(1, 20000),
        "vpn": random.choice([True, False]),
        "horario": random.uniform(0, 24),
        "tempo": random.uniform(0, 172800)  # range original do dataset
    }
"""
def gerar_transacao(fraudulenta=False):

    if not fraudulenta:
        distancia = random.uniform(0, 300)  
        vpn = False  
        horario = random.uniform(6, 23)  
    else:
        distancia = random.uniform(1000, 8000)
        vpn = random.choice([True, False])
        horario = random.uniform(0, 24)

        if random.random() < 0.5:  
            distancia = random.uniform(8000, 20000)
            vpn = True
            horario = random.uniform(0, 5)

    return {
        "fraudulenta": fraudulenta,
        "valor": round(random.uniform(1, 3000), 2),
        "distancia_km": distancia,
        "vpn": vpn,
        "horario": horario,
        "tempo": random.uniform(0, 172800)
    }

def transformar_para_pca(transacao):
    if transacao["fraudulenta"]:

        if transacao["vpn"] and transacao["distancia_km"] > 1000 and transacao["horario"] < 5:
            v = fraudes_reais[np.random.randint(0, len(fraudes_reais))]
        else:
            base = fraudes_reais[np.random.randint(0, len(fraudes_reais))]
            v = base + np.random.normal(0, 0.4, 28)

    else:
        v = np.random.normal(0, 0.4, 28)

    entrada = np.concatenate((
        [transacao["tempo"]],
        v,
        [transacao["valor"]]
    ))

    return entrada.reshape(1, -1)

def classificar(transacao):
    entrada = transformar_para_pca(transacao)
    entrada_scaled = scaler.transform(entrada)
    prob = modelo.predict_proba(entrada_scaled)[0][1]

    classe = modelo.predict(entrada_scaled)[0]
    return prob * 100, classe

def simular():
    print("\n=== SIMULADOR DE TRANSAÇÕES ===")
    print("1. Gerar transação legítima")
    print("2. Gerar fraude")

    opcao = input("Escolha: ")

    if opcao == "1":
        trans = gerar_transacao(fraudulenta=False)
    elif opcao == "2":
        trans = gerar_transacao(fraudulenta=True)
    else:
        return

    prob, classe = classificar(trans)

    print("\n=== RESULTADO ===")
    print(f"Valor da transação: R$ {trans['valor']}")
    print(f"Distância da última compra: {int(trans['distancia_km'])} km")
    print(f"VPN detectada: {'SIM' if trans['vpn'] else 'NÃO'}")
    print(f"Horário da compra: {round(trans['horario'], 1)}h")
    print(f"Probabilidade de fraude: {prob:.2f}%")
    print(f"Classificação: {'FRAUDE' if classe == 1 else 'Legítima'}")

while True:
    simular()
