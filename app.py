import streamlit as st

# Configuração da página DEVE SER A PRIMEIRA CHAMADA
st.set_page_config(
    page_title="Análise de Rotas Inteligente",
    layout="wide",
    page_icon="📊"
)

# Restante das importações
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# Configurações de compatibilidade do numpy (manter se for necessário no seu ambiente)
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

TIMEZONE = pytz.timezone('America/Sao_Paulo')

# Tema personalizado MELHORADO
custom_theme = """
<style>
:root {
    --primary-color: #00AFFF; /* Azul mais claro e vibrante */
    --background-color: #1E1E1E; /* Cinza escuro */
    --secondary-background-color: #2D2D2D; /* Cinza um pouco mais claro para sidebar/elementos */
    --accent-color: #FF4B4B; /* Vermelho para destaque/alertas */
    --text-color: #FFFFFF; /* Branco */
    --header-font: 'Segoe UI', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--header-font);
    color: var(--text-color);
    background-color: var(--background-color);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary-color);
    font-weight: 600;
    /* Ajustar cor dos headers dentro de expanders, se necessário */
}

/* Ajustar a cor do texto dentro de expanders para melhor contraste */
.stExpander {
    background-color: var(--secondary-background-color);
    padding: 10px; /* Adiciona um pouco de padding */
    border-radius: 8px;
    margin-bottom: 15px; /* Espaço entre expanders */
}

.stExpander > div > div > p {
     color: var(--text-color); /* Garante que o texto dentro do expander seja visível */
}

.stApp {
    background-color: var(--background-color);
    color: var(--text-color); /* Garante que o texto geral do app use a cor definida */
}

.stSidebar {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color);
}

.stSidebar .stMarkdown {
     color: var(--text-color); /* Garante que o markdown na sidebar use a cor definida */
}


.stButton>button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
    border: none; /* Remover borda padrão */
    padding: 10px 20px; /* Padding para melhor aparência */
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #0099E6; /* Cor um pouco mais escura no hover */
}

.stCheckbox>label {
    color: var(--text-color);
}

.stSelectbox>label {
    color: var(--text-color);
}

/* Melhorar aparência do date input */
.stDateInput > label {
    color: var(--text-color);
}

.stDateInput input {
    color: var(--text-color);
    background-color: var(--secondary-background-color);
    border: 1px solid #555; /* Borda sutil */
    border-radius: 4px;
    padding: 5px;
}

/* Melhorar aparência do slider */
.stSlider > label {
    color: var(--text-color);
}

.stSlider [data-baseweb="slider"] > div {
    background-color: var(--primary-color); /* Cor da barra preenchida */
}

.stSpinner > div > div {
    color: var(--primary-color); /* Cor do spinner */
}

</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

# Use st.secrets para credenciais de banco de dados
# Para configurar: crie um arquivo .streamlit/secrets.toml
# Exemplo:
# [mysql]
# host = "185.213.81.52"
# user = "u335174317_wazeportal"
# password = "@Ndre2025." # Mude isso para sua senha real ou use secrets
# database = "u335174317_wazeportal"

# faz conexxão com o banco de dados MySQL
@st.cache_resource # Cache a conexão do banco de dados
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.stop() # Parar a execução se não conseguir conectar

# Carregar apenas nomes das rotas (cached)
@st.cache_data(ttl=3600) # Cache por 1 hora
def get_all_route_names():
    mydb = None
    mycursor = None
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()
        query = "SELECT DISTINCT name FROM routes"
        mycursor.execute(query)
        results = mycursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        st.error(f"Erro ao obter nomes das rotas: {e}")
        return []
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão aqui se estiver usando @st.cache_resource
        # if mydb: mydb.close()


@st.cache_data(ttl=600) # Cache por 10 minutos, dados podem mudar mais frequentemente
def get_data(start_date=None, end_date=None, route_name=None):
    mydb = None
    mycursor = None
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()

        # Modificado para filtrar por nome da rota diretamente
        query = """
            SELECT hr.route_id, r.name AS route_name, hr.data, hr.velocidade
            FROM historic_routes hr
            JOIN routes r ON hr.route_id = r.id
        """
        conditions = []
        params = []

        if route_name:
             conditions.append("r.name = %s")
             params.append(route_name)
        if start_date:
            conditions.append("hr.data >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("hr.data <= %s")
            params.append(end_date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY hr.data ASC"

        mycursor.execute(query, params)
        results = mycursor.fetchall()
        col_names = [desc[0] for desc in mycursor.description]
        df = pd.DataFrame(results, columns=col_names)

        # Convertendo 'data' para datetime e 'velocidade' para numérico
        df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None)
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')

        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e) # Retorna DataFrame vazio e erro
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão aqui se estiver usando @st.cache_resource
        # if mydb: mydb.close()


@st.cache_data(ttl=3600) # Cache por 1 hora, coordenadas não mudam
def get_route_coordinates(route_id):
    mydb = None
    mycursor = None
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()
        query = "SELECT x, y FROM route_lines WHERE route_id = %s ORDER BY id"
        mycursor.execute(query, (route_id,))
        results = mycursor.fetchall()
        df = pd.DataFrame(results, columns=['longitude', 'latitude'])
        return df
    except Exception as e:
        st.error(f"Erro ao obter coordenadas: {e}")
        return pd.DataFrame()
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão aqui se estiver usando @st.cache_resource
        # if mydb: mydb.close()


def clean_data(df):
    # Assume que o DataFrame já está filtrado pela rota e período
    df = df.copy()
    # df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None) # Já feito em get_data
    df = df.sort_values('data')
    df['velocidade'] = (
        df['velocidade']
        .clip(upper=150)
        .interpolate(method='linear')
        .ffill()
        .bfill()
    )
    df['day_of_week'] = df['data'].dt.day_name()
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade'])

def detect_anomalies(df):
    # Esta função não está sendo usada no main, mas mantida e revisada
    df = df.copy()
    if len(df) < 2: # Precisa de pelo menos 2 pontos para calcular a diferença
        return pd.DataFrame()
    df['vel_diff'] = df['velocidade'].diff().abs()
    # Aumentar o multiplicador para tornar a detecção de anomalias menos sensível a pequenas variações
    threshold = df['vel_diff'].quantile(0.98) * 2 # Ajuste o multiplicador conforme necessário
    # Garantir um limite mínimo para evitar detectar ruído em dados muito estáveis
    min_threshold = 10 # Ex: 10 km/h de diferença instantânea
    final_threshold = max(threshold, min_threshold)

    # Filtra pontos onde a diferença é maior que o limite E a velocidade em si parece incomum
    # (ex: velocidade muito baixa ou muito alta comparada à média)
    anomalies = df[
        (df['vel_diff'] > final_threshold) &
        (
            (df['velocidade'] < df['velocidade'].mean() * 0.5) | # Exemplo: velocidade < 50% da média
            (df['velocidade'] > df['velocidade'].mean() * 1.5)   # Exemplo: velocidade > 150% da média
        )
    ].copy()
    return anomalies


# Função de decomposição sazonal (revisada para usar índice de tempo)
def seasonal_decomposition_plot(df):
    if len(df) < 2 * 480: # Precisa de pelo menos 2 períodos para decomposição sazonal com period=480
         st.warning("Dados insuficientes para decomposição sazonal. Necessário pelo menos 16 horas de dados (aprox. 2*480 pontos).")
         return

    # Garantir frequência temporal, interpolando se houver lacunas
    df_ts = df.set_index('data')['velocidade'].asfreq('3min').interpolate(method='time')

    # Verifique se a interpolação resultou em dados suficientes para a decomposição
    if len(df_ts.dropna()) < 2 * 480:
         st.warning("Dados insuficientes (mesmo após interpolação) para decomposição sazonal.")
         return

    try:
        # period=480 para sazonalidade diária em dados de 3 em 3 minutos (24*60/3 = 480)
        decomposition = seasonal_decompose(df_ts.dropna(), model='additive', period=480)
        fig, ax = plt.subplots(4, 1, figsize=(12, 10)) # Adiciona componente de Resíduo
        decomposition.observed.plot(ax=ax[0], title='Observado')
        decomposition.trend.plot(ax=ax[1], title='Tendência')
        decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade (Periodo 480)')
        decomposition.resid.plot(ax=ax[3], title='Resíduo')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
         st.warning(f"Não foi possível realizar a decomposição sazonal: {e}")


# Função de previsão ARIMA (revisada para usar intervalos de confiança)
@st.cache_data(ttl=300) # Cache por 5 minutos para previsões
def create_arima_forecast(df, route_id, steps=10):
    if len(df) < 24 * 2 * 4: # Precisa de dados suficientes para o auto_arima com sazonalidade m=480
         st.warning("Dados insuficientes para o modelo de previsão. Necessário mais dados históricos.")
         return pd.DataFrame()

    # Preparar dados para auto_arima (já vem limpo e com índice de tempo 'ds' e valor 'y')
    arima_data = df.set_index('data')['velocidade'].asfreq('3min').dropna()


    if len(arima_data) < 2:
        st.warning("Dados insuficientes após interpolação/limpeza para rodar o ARIMA.")
        return pd.DataFrame()

    try:
        # auto_arima encontrará os melhores parâmetros p,d,q,P,D,Q
        # m=480 para sazonalidade diária em dados de 3 em 3 minutos
        model = auto_arima(arima_data, seasonal=True, m=480,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True, random_state=42, n_periods=steps, # Adicionado stepwise e random_state
                           n_fits=10) # Limitar o número de fits para evitar tempo excessivo

        # Realizar a previsão com intervalos de confiança
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)

        last_date = arima_data.index.max()
        # Gerar datas futuras com base na última data e frequência de 3 minutos
        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='3min')[1:]

        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast,
            'yhat_lower': conf_int[:, 0], # Limite inferior do intervalo de confiança
            'yhat_upper': conf_int[:, 1], # Limite superior do intervalo de confiança
            'id_route': route_id
        })

        # Garante que as previsões e intervalos de confiança não são negativos
        forecast_df[['yhat', 'yhat_lower', 'yhat_upper']] = forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

        return forecast_df
    except Exception as e:
        st.error(f"Erro ao gerar modelo de previsão ou prever: {str(e)}")
        st.info("Tente selecionar um período de dados maior ou mais estável para a previsão.")
        return pd.DataFrame()

# Não cachear a função de salvar no DB
def save_forecast_to_db(forecast_df):
    if forecast_df.empty:
        return # Não salva se o DataFrame estiver vazio
    try:
        # Usando credenciais do secrets
        engine = create_engine(f'mysql+mysqlconnector://{st.secrets["mysql"]["user"]}:{st.secrets["mysql"]["password"]}@{st.secrets["mysql"]["host"]}/{st.secrets["mysql"]["database"]}')
        with engine.begin() as connection:
             # Mapear nomes de colunas do DataFrame para nomes de colunas da tabela forecast_history se forem diferentes
             # Assumindo que a tabela forecast_history tem colunas como 'data', 'previsao', 'limite_inferior', 'limite_superior', 'id_rota'
             # Ajuste conforme a estrutura real da sua tabela 'forecast_history'
             forecast_df_mapped = forecast_df.rename(columns={
                 'ds': 'data',
                 'yhat': 'previsao',
                 'yhat_lower': 'limite_inferior',
                 'yhat_upper': 'limite_superior',
                 'id_route': 'id_rota'
             })
             forecast_df_mapped.to_sql('forecast_history', con=connection, if_exists='append', index=False)
             st.success("Previsão salva no banco de dados!") # Feedback ao usuário
    except Exception as e:
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")


def gerar_insights(df):
    insights = []
    if df.empty:
        return "Não há dados para gerar insights neste período."

    media_geral = df['velocidade'].mean()
    insights.append(f"📌 Velocidade média geral: **{media_geral:.2f} km/h**")

    # Encontrar o dia e hora mais lentos dentro do período selecionado
    daily_avg = df.groupby(df['data'].dt.date)['velocidade'].mean()
    if not daily_avg.empty:
        dia_mais_lento_date = daily_avg.idxmin()
        velocidade_dia_mais_lento = daily_avg.min()
        insights.append(f"📅 Dia com a menor velocidade média: **{dia_mais_lento_date}** ({velocidade_dia_mais_lento:.2f} km/h)")

    weekday_avg = df.groupby('day_of_week')['velocidade'].mean()
    if not weekday_avg.empty:
        # Reordenar para encontrar o dia da semana mais lento na ordem correta
        dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_avg = weekday_avg.reindex(dias_ordenados)
        dia_da_semana_mais_lento = weekday_avg.idxmin()
        insights.append(f"📅 Dia da semana mais lento (em média): **{dia_da_semana_mais_lento}**")

    hourly_avg = df.groupby('hour')['velocidade'].mean()
    if not hourly_avg.empty:
        hora_mais_lenta = hourly_avg.idxmin()
        insights.append(f"🕒 Hora do dia mais lenta (em média): **{hora_mais_lenta:02d}:00**")

    return "\n\n".join(insights)


def main():
    # Verificar se as secrets do banco de dados estão configuradas
    if "mysql" not in st.secrets:
        st.error("As credenciais do banco de dados não foram configuradas no secrets.toml.")
        st.markdown("Por favor, crie ou atualize o arquivo `.streamlit/secrets.toml` com as informações de conexão do MySQL.")
        st.stop() # Parar a execução

    with st.sidebar:
        st.title("ℹ️ Painel de Controle")
        st.markdown("""
            Configure a análise de rotas aqui.

            **Configurações Avançadas:**
            - Compare múltiplas rotas
            - Ajuste o período de análise
            - Altere o horizonte de previsão
        """)

        st.subheader("Seleção de Rotas")
        # Carregar nomes das rotas de forma eficiente
        all_route_names = get_all_route_names()
        if not all_route_names:
             st.warning("Não foi possível carregar os nomes das rotas do banco de dados.")
             st.stop() # Parar se não houver rotas

        route_name = st.selectbox("Rota Principal:", all_route_names, key="main_route_select")

        compare_enabled = st.checkbox("Comparar com outra rota", key="compare_checkbox")
        second_route = None
        if compare_enabled:
            available_for_comparison = [r for r in all_route_names if r != route_name]
            if available_for_comparison:
                 second_route = st.selectbox("Rota Secundária:", available_for_comparison, key="secondary_route_select")
            else:
                 st.warning("Não há outras rotas disponíveis para comparação.")
                 compare_enabled = False # Desabilita comparação se não houver outras rotas


        st.subheader("Período de Análise")
        # Usar um seletor de data único ou por rota? Manter por rota por enquanto.
        date_range_main = st.date_input(
            f"Intervalo para {route_name}",
            value=(pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')),
            max_value=pd.to_datetime('today'),
            key=f"date_range_{route_name}"
        )

        date_range_secondary = None
        if compare_enabled and second_route:
             date_range_secondary = st.date_input(
                f"Intervalo para {second_route}",
                value=(pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')),
                max_value=pd.to_datetime('today'),
                key=f"date_range_{second_route}"
            )
             if date_range_secondary[0] > date_range_secondary[1]:
                 st.error("Data final da rota secundária não pode ser anterior à data inicial.")
                 st.stop()


        if date_range_main[0] > date_range_main[1]:
            st.error("Data final da rota principal não pode ser anterior à data inicial")
            st.stop()

    st.title("🚀 Análise de Rotas Inteligente")

    routes_info = {}
    routes_to_process = [route_name]
    if compare_enabled and second_route:
        routes_to_process.append(second_route)

    # --- Carregamento e Processamento de Dados ---
    st.header("⏳ Processando Dados...")
    for route in routes_to_process:
        date_range = date_range_main if route == route_name else date_range_secondary
        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')

        with st.spinner(f'Carregando e processando dados para {route}...'):
            # Carregar dados filtrando por nome da rota e período
            raw_df, error = get_data(
                start_date=start_date,
                end_date=end_date,
                route_name=route
            )

            if error:
                st.error(f"Erro ao carregar dados para {route}: {error}")
                routes_info[route] = {'data': pd.DataFrame(), 'id': None, 'error': error}
                continue # Pula para a próxima rota se houver erro

            if raw_df.empty:
                st.warning(f"Nenhum dado encontrado para a rota '{route}' no período selecionado.")
                routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                continue # Pula para a próxima rota

            # Obter o ID da rota (assumindo que há apenas um ID por nome no período selecionado)
            route_id = raw_df['route_id'].iloc[0]

            # Limpar e processar os dados
            processed_df = clean_data(raw_df)

            routes_info[route] = {
                'data': processed_df,
                'id': route_id
            }
        st.success(f"Dados para {route} carregados e processados.")

    # --- Seção do Mapa ---
    st.header("🗺️ Visualização Geográfica")
    for route in routes_to_process:
        if route in routes_info and not routes_info[route]['data'].empty:
            route_id = routes_info[route]['id']
            with st.expander(f"Mapa da Rota: {route}", expanded=True):
                route_coords = get_route_coordinates(route_id)

                if not route_coords.empty:
                    # Calcular bounds para centralizar o mapa
                    min_lat, max_lat = route_coords['latitude'].min(), route_coords['latitude'].max()
                    min_lon, max_lon = route_coords['longitude'].min(), route_coords['longitude'].max()

                    # Adicionar um pequeno buffer
                    lat_buffer = (max_lat - min_lat) * 0.1
                    lon_buffer = (max_lon - min_lon) * 0.1

                    center_lat = (max_lat + min_lat) / 2
                    center_lon = (max_lon + min_lon) / 2

                    # Ajustar zoom inicial e bounds
                    fig = go.Figure(go.Scattermapbox(
                        mode="lines+markers",
                        lon=route_coords['longitude'],
                        lat=route_coords['latitude'],
                        marker={'size': 8, 'color': var(--accent-color)}, # Usando cor do tema
                        line=dict(width=4, color=var(--primary-color)), # Usando cor do tema
                        hovertext=[f"Ponto {i+1}" for i in range(len(route_coords))],
                        hoverinfo="text+lat+lon"
                    ))

                    fig.update_layout(
                        mapbox={
                            'style': "carto-darkmatter", # Estilo de mapa que combina com o tema escuro
                            'center': {'lat': center_lat, 'lon': center_lon},
                            'zoom': 12, # Zoom inicial, pode ser ajustado
                            # Bounds podem ajudar a focar na área, mas 'center' e 'zoom' são mais comuns
                            # 'bounds': {'west': min_lon - lon_buffer, 'east': max_lon + lon_buffer,
                            #            'south': min_lat - lat_buffer, 'north': max_lat + lat_buffer}
                        },
                        margin={"r":0,"t":0,"l":0,"b":0},
                        height=500 # Altura do mapa
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Nenhuma coordenada geográfica encontrada para a rota '{route}'.")
        elif route in routes_info and 'error' in routes_info[route]:
             st.warning(f"Mapa não disponível para '{route}' devido a erro no carregamento de dados.")


    # --- Seção de Análise ---
    st.header("📈 Análise Preditiva")
    for route in routes_to_process:
        if route in routes_info and not routes_info[route]['data'].empty:
            processed_df = routes_info[route]['data']
            route_id = routes_info[route]['id']

            with st.expander(f"Análise para {route}", expanded=True):

                st.subheader("🧠 Insights Automáticos")
                st.markdown(gerar_insights(processed_df))

                st.subheader("📉 Decomposição Temporal")
                # Passa o df processado que clean_data retornou
                seasonal_decomposition_plot(processed_df)

                st.subheader("🔥 Heatmap Horário por Dia da Semana")
                if not processed_df.empty:
                    pivot_table = processed_df.pivot_table(
                        index='day_of_week',
                        columns='hour',
                        values='velocidade',
                        aggfunc='mean'
                    )

                    # Reordenar dias da semana (em português se preferir, mas mantive inglês para o código)
                    dias_ordenados_eng = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    # Mapeamento para português se quiser exibir no gráfico
                    dias_pt = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                    dia_mapping = dict(zip(dias_ordenados_eng, dias_pt))

                    # Reindexar a tabela pivotada
                    pivot_table = pivot_table.reindex(dias_ordenados_eng)
                    pivot_table.index = pivot_table.index.map(dia_mapping) # Renomear índice para português

                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Usar cmap que funcione bem em fundo escuro
                    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=ax) # 'viridis' ou 'plasma' ou 'cividis'
                    ax.set_title("Velocidade Média por Dia da Semana e Hora", color=var(--text-color))
                    ax.set_xlabel("Hora do Dia", color=var(--text-color))
                    ax.set_ylabel("Dia da Semana", color=var(--text-color))
                    # Ajustar cor dos ticks e labels
                    ax.tick_params(axis='x', colors=var(--text-color))
                    ax.tick_params(axis='y', colors=var(--text-color))
                    # Mudar cor do background do plot
                    fig.patch.set_facecolor(var(--secondary-background-color))
                    ax.set_facecolor(var(--secondary-background-color))
                    st.pyplot(fig)
                else:
                     st.info("Dados insuficientes para gerar o Heatmap.")


                st.subheader("🔮 Previsão de Velocidade (Modelo ARIMA)")
                # Certificar que o DataFrame tem dados e foi limpo
                if not processed_df.empty:
                     # A frequência da série temporal é 3 minutos, baseada na coleta
                     steps = st.slider(f"⏱️ Passos de previsão para {route} (3min cada)", 5, 60, 10, key=f"steps_{route}") # Aumentei o max slider
                     arima_forecast = create_arima_forecast(processed_df, route_id, steps)

                     if not arima_forecast.empty:
                         fig = go.Figure()

                         # Adicionar dados históricos
                         fig.add_trace(go.Scatter(
                             x=processed_df['data'],
                             y=processed_df['velocidade'],
                             mode='lines',
                             name='Histórico',
                             line=dict(color=var(--primary-color), width=2)
                         ))

                         # Adicionar previsão
                         fig.add_trace(go.Scatter(
                             x=arima_forecast['ds'],
                             y=arima_forecast['yhat'],
                             mode='lines',
                             name='Previsão',
                             line=dict(color=var(--accent-color), width=2, dash='dash')
                         ))

                         # Adicionar intervalo de confiança
                         fig.add_trace(go.Scatter(
                             x=arima_forecast['ds'].tolist() + arima_forecast['ds'][::-1].tolist(), # Datas para preencher a área
                             y=arima_forecast['yhat_upper'].tolist() + arima_forecast['yhat_lower'][::-1].tolist(), # Limites para preencher a área
                             fill='toself',
                             fillcolor='rgba(0, 175, 255, 0.2)', # Cor semi-transparente (azul claro)
                             line=dict(color='rgba(255,255,255,0)'), # Linha transparente
                             name='Intervalo de Confiança (95%)',
                             showlegend=True
                         ))

                         fig.update_layout(
                             title=f"Histórico e Previsão para {route}",
                             xaxis_title="Data e Hora",
                             yaxis_title="Velocidade (km/h)",
                             hovermode='x unified', # Melhorar hover
                             plot_bgcolor=var(--secondary-background-color), # Fundo do plot
                             paper_bgcolor=var(--secondary-background-color), # Fundo do paper/fora do plot
                             font=dict(color=var(--text-color)), # Cor da fonte geral do gráfico
                             xaxis=dict(showgrid=True, gridcolor='#555'), # Grid sutil
                             yaxis=dict(showgrid=True, gridcolor='#555'), # Grid sutil
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Posicionar legenda
                         )
                         st.plotly_chart(fig)

                         # Botão para salvar a previsão
                         if st.button(f"💾 Salvar Última Previsão para {route}", key=f"save_forecast_{route}"):
                              save_forecast_to_db(arima_forecast)

                     else:
                         st.info("Não foi possível gerar a previsão. Verifique se há dados suficientes para o período selecionado.")
                else:
                     st.info("Dados insuficientes para rodar a análise de previsão para esta rota e período.")

        elif route in routes_info and 'error' in routes_info[route]:
             st.warning(f"Análise preditiva não disponível para '{route}' devido a erro no carregamento de dados.")


    # --- Seção Técnica ---
    st.header("⚙️ Detalhes Técnicos")
    with st.expander("Relatório de Qualidade de Dados"):
        for route in routes_to_process:
            if route in routes_info and not routes_info[route]['data'].empty:
                st.subheader(f"Qualidade dos Dados: {route}")
                processed_df = routes_info[route]['data']
                report = {
                    "total_registros": len(processed_df),
                    "registros_velocidade_nulos_apos_limpeza": processed_df['velocidade'].isnull().sum(), # Deve ser 0 se o ffill/bfill funcionou
                    # anomaly detection needs to be implemented and used if desired in the report
                    # "outliers_detectados": len(detect_anomalies(processed_df)), # Usar se a detecção de anomalias for usada
                    "cobertura_temporal": f"{processed_df['data'].min().strftime('%Y-%m-%d %H:%M')} a {processed_df['data'].max().strftime('%Y-%m-%d %H:%M')}"
                }
                st.json(report)
            elif route in routes_info and 'error' in routes_info[route]:
                 st.warning(f"Relatório de qualidade não disponível para '{route}' devido a erro no carregamento de dados.")


if __name__ == "__main__":
    main()