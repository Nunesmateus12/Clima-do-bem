# Clima-do-bemimport numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Simulação de Dados Agrícolas (Representando sensores reais)
# No mundo real, esses dados viriam de APIs como INMET ou sensores IoT.
def gerar_dados_agricolas(n_amostras=1000):
    np.random.seed(42)
    
    # Recursos (Features)
    temperatura = np.random.uniform(15, 40, n_amostras)       # em °C
    umidade_ar = np.random.uniform(30, 90, n_amostras)         # em %
    umidade_solo = np.random.uniform(10, 80, n_amostras)       # em %
    precipitacao_prevista = np.random.uniform(0, 50, n_amostras) # em mm
    
    df = pd.DataFrame({
        'temperatura_C': temperatura,
        'umidade_ar_pct': umidade_ar,
        'umidade_solo_pct': umidade_solo,
        'chuva_prevista_mm': precipitacao_prevista
    })
    
    # Regra de negócio lógica para a IA aprender:
    # Precisa irrigar (1) se o solo estiver seco (< 30%) E não houver previsão de chuva forte (< 10mm)
    df['necessita_irrigacao'] = np.where(
        (df['umidade_solo_pct'] < 35) & (df['chuva_prevista_mm'] < 15), 1, 0
    )
    
    return df

# 2. Preparação dos Dados
print("🌾 Gerando base de dados sintética do AgroBalance...")
dados = gerar_dados_agricolas()

X = dados[['temperatura_C', 'umidade_ar_pct', 'umidade_solo_pct', 'chuva_prevista_mm']]
y = dados['necessita_irrigacao']

# Divisão em Treino (80%) e Teste (20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinamento da Inteligência Artificial
print("🧠 Treinando o modelo de Machine Learning (Random Forest)...")
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_treino, y_treino)

# 4. Avaliação do Modelo
predicoes = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, predicoes)

print("\n📊 --- RESULTADOS DO MODELO ---")
print(f"Acurácia Geral: {acuracia * 100:.2f}%")
print("\nRelatório de Classificação:")
print(classification_report(y_teste, predicoes, target_names=['Não Irrigar', 'Irrigar']))

# 5. Teste Prático (Simulação de uma consulta do agricultor)
print("\n🔮 Testando predição com novos dados de sensores:")
novo_cenario = pd.DataFrame([{
    'temperatura_C': 32.5,
    'umidade_ar_pct': 40.0,
    'umidade_solo_pct': 22.0,  # Solo bem seco
    'chuva_prevista_mm': 2.0    # Quase nenhuma chuva prevista
}])

resultado = modelo.predict(novo_cenario)
decisao = "LIGAR IRRIGACAO" if resultado[0] == 1 else "MANTER DESLIGADO (Economia de Água)"
print(f"Status dos Sensores -> Solo: 22%, Chuva: 2mm. Decisão do AgroBalance: **{decisao}**")
