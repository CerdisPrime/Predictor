# Content

This dataset includes **49,398 results of international football matches starting from the very first official match in 1872 up to 2024. The matches range from FIFA World Cup to FIFI Wild Cup to regular friendly matches. The matches are strictly men's full internationals and the data does not include Olympic Games or matches where at least one of the teams was the nation's B-team, U-23 or a league select team.

`results.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `home_score` - full-time home team score including extra time, not including penalty-shootouts
-   `away_score` - full-time away team score including extra time, not including penalty-shootouts
-   `tournament` - the name of the tournament
-   `city` - the name of the city/town/administrative unit where the match was played
-   `country` - the name of the country where the match was played
-   `neutral` - TRUE/FALSE column indicating whether the match was played at a neutral venue

`shootouts.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `winner` - winner of the penalty-shootout
-   `first_shooter` - the team that went first in the shootout

`goalscorers.csv` includes the following columns:

-   `date` - date of the match
-   `home_team` - the name of the home team
-   `away_team` - the name of the away team
-   `team` - name of the team scoring the goal
-   `scorer` - name of the player scoring the goal
-   `own_goal` - whether the goal was an own-goal
-   `penalty` - whether the goal was a penalty

Note on team and country names: For home and away teams the *current* name of the team has been used. For example, when in 1882 a team who called themselves Ireland played against England, in this dataset, it is called Northern Ireland because the current team of Northern Ireland is the successor of the 1882 Ireland team. This is done so it is easier to track the history and statistics of teams.

For country names, the name of the country *at the time of the match* is used. So when Ghana played in Accra, Gold Coast in the 1950s, even though the names of the home team and the country don't match, it was a home match for Ghana. This is indicated by the neutral column, which says FALSE for those matches, meaning it was **not** at a neutral venue.

# Simulador
Rol: Eres un Científico de Datos especializado en analítica de fútbol de élite. 
Te he adjuntado tres archivos: la base de datos results.csv, el diccionario de nombres former_names.csv y un script de Python de mi autoría que implementa un ensamble predictivo avanzado (Hybrid Random Forest + MCMC Bayesian Mixture).
### Instrucciones de Ejecución:
1. Configura el entorno: Asegúrate de tener los archivos CSV en el mismo directorio de ejecución que el script.
2. Modifica las variables iniciales: Dentro del código de Python, cambia el valor de las variables EQUIPO_LOCAL y EQUIPO_VISITANTE por las selecciones que te indicaré al final de este prompt.
3. Ejecuta el código: Corre el script completo en tu entorno interno de Python. El código ya se encarga de aplicar los cruces de habilidad, entrenar los 5000 árboles para el bosque aleatorio y generar el muestreo con la restricción de suma cero para la mezcla bayesiana.
4. Captura los resultados.
### Formato de Salida:
No me expliques el código, no modifiques la metodología y no me narres el proceso. Limítate a ejecutar el script y entregarme textualmente el Reporte Predictivo Avanzado que imprime la consola con las siguientes métricas exactas:
1. Tasas Esperadas: Los valores de Lambda ($\lambda$) resultantes del ensamble para cada equipo.
2. Mercado 1X2: Porcentajes exactos de Victoria Local, Empate y Victoria Visitante.
3. Top 5 Marcadores Exactos: Los resultados con mayor probabilidad porcentual.
4. Goles Totales: Probabilidad de Ambos Anotan (BTTS) y de la línea de Más/Menos de 1.5, 2.5 y 3.5 Goles (Over/Under).
