import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import RandomForestRegressor
import pymc as pm
import warnings

# Ignorar advertencias menores para un output más limpio
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN DEL PARTIDO A SIMULAR
# ==========================================
EQUIPO_LOCAL = 'Mexico'
EQUIPO_VISITANTE = 'Germany'
FECHA_CORTE = '2018-01-01' # Ventana de datos (últimos ciclos mundialistas)

print(f"Iniciando Ensamble Algorítmico: {EQUIPO_LOCAL} vs {EQUIPO_VISITANTE}...")

# ==========================================
# 1. PREPARACIÓN Y LIMPIEZA DE DATOS
# ==========================================
print("\n[1/4] Cargando y procesando bases de datos...")
try:
    results = pd.read_csv('results.csv')
    former_names = pd.read_csv('former_names.csv')
except FileNotFoundError:
    print("Error: Asegúrate de que 'results.csv' y 'former_names.csv' estén en la misma carpeta que este script.")
    exit()

# Homogeneizar nombres
name_mapping = dict(zip(former_names['former_name'], former_names['current_name']))
results['home_team'] = results['home_team'].replace(name_mapping)
results['away_team'] = results['away_team'].replace(name_mapping)

# Filtrar ventana temporal
results['date'] = pd.to_datetime(results['date'])
df = results[results['date'] >= FECHA_CORTE].copy()

# ==========================================
# 2. MODELO 1: HYBRID RANDOM FOREST (Groll et al., 2019)
# ==========================================
print("[2/4] Entrenando Hybrid Random Forest (Groll et al. 2019)...")

# A. Calcular "Parámetros de Habilidad" históricos (Base del modelo Poisson para inyectar al RF)
avg_global = (df['home_score'].mean() + df['away_score'].mean()) / 2

# Fuerzas promedio por equipo
home_perf = df.groupby('home_team')[['home_score', 'away_score']].mean()
away_perf = df.groupby('away_team')[['away_score', 'home_score']].mean()

teams = set(df['home_team']).union(set(df['away_team']))
habilidades = {}
for t in teams:
    att_h = home_perf.loc[t, 'home_score'] if t in home_perf.index else avg_global
    def_h = home_perf.loc[t, 'away_score'] if t in home_perf.index else avg_global
    att_a = away_perf.loc[t, 'away_score'] if t in away_perf.index else avg_global
    def_a = away_perf.loc[t, 'home_score'] if t in away_perf.index else avg_global
    
    habilidades[t] = {
        'att_ability': ((att_h + att_a) / 2) / avg_global,
        'def_ability': ((def_h + def_a) / 2) / avg_global
    }

# B. Inyectar habilidades como covariables (Features)
df['home_att'] = df['home_team'].map(lambda x: habilidades[x]['att_ability'])
df['home_def'] = df['home_team'].map(lambda x: habilidades[x]['def_ability'])
df['away_att'] = df['away_team'].map(lambda x: habilidades[x]['att_ability'])
df['away_def'] = df['away_team'].map(lambda x: habilidades[x]['def_ability'])
df['is_neutral'] = df['neutral'].astype(int)

features = ['home_att', 'home_def', 'away_att', 'away_def', 'is_neutral']
X = df[features].fillna(1.0)
y_home = df['home_score']
y_away = df['away_score']

# C. Entrenar el Bosque Aleatorio (B=5000 como indica el paper)
rf_home = RandomForestRegressor(n_estimators=5000, max_features='sqrt', random_state=42, n_jobs=-1)
rf_away = RandomForestRegressor(n_estimators=5000, max_features='sqrt', random_state=42, n_jobs=-1)

rf_home.fit(X, y_home)
rf_away.fit(X, y_away)

# Predicción del RF para el partido específico
X_partido = pd.DataFrame([{
    'home_att': habilidades[EQUIPO_LOCAL]['att_ability'],
    'home_def': habilidades[EQUIPO_LOCAL]['def_ability'],
    'away_att': habilidades[EQUIPO_VISITANTE]['att_ability'],
    'away_def': habilidades[EQUIPO_VISITANTE]['def_ability'],
    'is_neutral': 1 # Sede neutral (Mundial)
}])

lambda_rf_home = rf_home.predict(X_partido)[0]
lambda_rf_away = rf_away.predict(X_partido)[0]

# ==========================================
# 3. MODELO 2: MCMC BAYESIAN MIXTURE (Baio & Blangiardo, 2010)
# ==========================================
print("[3/4] Ejecutando MCMC Bayesiano con Mixture Model (Baio & Blangiardo 2010)...")
print("      (Esto puede tardar un par de minutos debido al muestreo probabilístico)")

# Mapeo de equipos a índices para PyMC
lista_equipos = sorted(list(teams))
team_to_idx = {team: i for i, team in enumerate(lista_equipos)}
num_teams = len(lista_equipos)

idx_home = df['home_team'].map(team_to_idx).values
idx_away = df['away_team'].map(team_to_idx).values
obs_home = df['home_score'].values
obs_away = df['away_score'].values

# Modelo PyMC
with pm.Model() as baio_mixture_model:
    # 1. Pesos de los 3 grupos (Fondo, Media tabla, Top)
    w = pm.Dirichlet('w', a=np.ones(3))
    
    # 2. Medias de los grupos para Ataque (Malo=-1, Medio=0, Top=1) y Defensa (Top=-1, Medio=0, Malo=1)
    mu_att_comp = pm.Normal('mu_att_comp', mu=np.array([-1.0, 0.0, 1.0]), sigma=0.5, shape=3)
    mu_def_comp = pm.Normal('mu_def_comp', mu=np.array([1.0, 0.0, -1.0]), sigma=0.5, shape=3)
    
    # 3. Componente de Mezcla (Mixture) para clasificar a los equipos y evitar Overshrinkage
    atts = pm.Mixture('atts', w=w, comp_dists=pm.Normal.dist(mu=mu_att_comp, sigma=0.5), shape=num_teams)
    defs = pm.Mixture('defs', w=w, comp_dists=pm.Normal.dist(mu=mu_def_comp, sigma=0.5), shape=num_teams)
    
    # 4. Restricción suma-cero
    atts_star = atts - pm.math.mean(atts)
    defs_star = defs - pm.math.mean(defs)
    
    # 5. Ecuación log-lineal (Sede neutral, home_advantage = 0)
    theta_home = pm.math.exp(atts_star[idx_home] + defs_star[idx_away])
    theta_away = pm.math.exp(atts_star[idx_away] + defs_star[idx_home])
    
    # 6. Verosimilitud Poisson
    home_goals = pm.Poisson('home_goals', mu=theta_home, observed=obs_home)
    away_goals = pm.Poisson('away_goals', mu=theta_away, observed=obs_away)
    
    # Ejecutar muestreo (reducido para velocidad, subir tune/draws para más precisión)
    trace = pm.sample(1000, tune=500, target_accept=0.9, return_inferencedata=False, progressbar=False)

# Extraer parámetros del partido
idx_eq1 = team_to_idx[EQUIPO_LOCAL]
idx_eq2 = team_to_idx[EQUIPO_VISITANTE]

# Media posterior de las habilidades (Exp para obtener las tasas lambda en sede neutral)
lambda_bayes_home = np.exp(trace['atts'][:, idx_eq1].mean() + trace['defs'][:, idx_eq2].mean())
lambda_bayes_away = np.exp(trace['atts'][:, idx_eq2].mean() + trace['defs'][:, idx_eq1].mean())


# ==========================================
# 4. ENSAMBLE Y CÁLCULO DE PROBABILIDADES
# ==========================================
print("\n[4/4] Ensamblando modelos y calculando probabilidades combinadas...")

# Promedio ponderado (Ensamble 50/50 de ambos papers)
lambda_final_home = (lambda_rf_home + lambda_bayes_home) / 2
lambda_final_away = (lambda_rf_away + lambda_bayes_away) / 2

# Matriz de Distribución de Poisson Bivariada (Aproximación Independiente)
max_goles = 7
matriz = np.zeros((max_goles, max_goles))
for i in range(max_goles):
    for j in range(max_goles):
        matriz[i, j] = poisson.pmf(i, lambda_final_home) * poisson.pmf(j, lambda_final_away)

# Extracción de Métricas de Valor
prob_local = np.sum(np.tril(matriz, -1)) * 100
prob_empate = np.sum(np.diag(matriz)) * 100
prob_visita = np.sum(np.triu(matriz, 1)) * 100

# Ambos Anotan (BTTS) -> Suma de la matriz excluyendo fila 0 y columna 0
prob_btts = np.sum(matriz[1:, 1:]) * 100

# Over/Under 2.5 Goles -> Suma de celdas donde i + j > 2
prob_over_25 = sum(matriz[i, j] for i in range(max_goles) for j in range(max_goles) if i + j > 2) * 100
prob_under_25 = 100 - prob_over_25

# Top 5 Marcadores
marcadores = []
for i in range(max_goles):
    for j in range(max_goles):
        marcadores.append((f"{i} - {j}", matriz[i, j] * 100))
marcadores.sort(key=lambda x: x[1], reverse=True)
top_5 = marcadores[:5]

# ==========================================
# REPORTE FINAL
# ==========================================
print("="*50)
print(f"REPORTE PREDICTIVO AVANZADO: {EQUIPO_LOCAL} vs {EQUIPO_VISITANTE}")
print("Metodología: Hybrid Random Forest + MCMC Bayesian Mixture")
print("="*50)
print(f"Tasas Esperadas (Ensamble):")
print(f"{EQUIPO_LOCAL} (λ): {lambda_final_home:.2f} | {EQUIPO_VISITANTE} (λ): {lambda_final_away:.2f}\n")

print("▶ MERCADO 1X2:")
print(f"  Victoria {EQUIPO_LOCAL}: {prob_local:.2f}%")
print(f"  Empate: {prob_empate:.2f}%")
print(f"  Victoria {EQUIPO_VISITANTE}: {prob_visita:.2f}%\n")

print("▶ TOP 5 MARCADORES EXACTOS:")
for idx, (marcador, prob) in enumerate(top_5, 1):
    print(f"  {idx}. [{EQUIPO_LOCAL} {marcador.split(' - ')[0]} - {marcador.split(' - ')[1]} {EQUIPO_VISITANTE}] : {prob:.2f}%")

print("\n▶ GOLES TOTALES:")
print(f"  Ambos Equipos Anotan (BTTS): {prob_btts:.2f}%")
print(f"  Más de 2.5 Goles (Over): {prob_over_25:.2f}%")
print(f"  Menos de 2.5 Goles (Under): {prob_under_25:.2f}%")
print("="*50)