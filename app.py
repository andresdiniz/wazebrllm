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

# Não use este TIMEZONE diretamente para localizar o DF, use .dt.tz_localize(None)
# e .dt.tz_convert se precisar de timezone aware
TIMEZONE = pytz.timezone('America/Sao_Paulo')

# Tema personalizado MELHORADO E COERENTE COM FUNDO ESCURO
# Definindo as cores em variáveis para fácil referência
PRIMARY_COLOR = "#00AFFF"        # Azul mais claro e vibrante
BACKGROUND_COLOR = "#1E1E1E"     # Cinza escuro para o fundo principal
SECONDARY_BACKGROUND_COLOR = "#2D2D2D" # Cinza um pouco mais claro para sidebar/elementos
ACCENT_COLOR = "#FF4B4B"         # Vermelho para destaque/alertas
TEXT_COLOR = "#FFFFFF"           # Branco
HEADER_FONT = 'Segoe UI', 'sans-serif' # Fonte

custom_theme = f"""
<style>
:root {{
    --primary-color: {PRIMARY_COLOR};
    --background-color: {BACKGROUND_COLOR};
    --secondary-background-color: {SECONDARY_BACKGROUND_COLOR};
    --accent-color: {ACCENT_COLOR};
    --text-color: {TEXT_COLOR};
    --header-font: {', '.join(HEADER_FONT)};
}}

html, body, [class*="css"] {{
    font-family: var(--header-font);
    color: var(--text-color);
    background-color: var(--background-color);
}}

h1, h2, h3, h4, h5, h6 {{
    color: var(--primary-color);
    font-weight: 600;
}}

/* Ajustar a cor do texto dentro de expanders para melhor contraste */
.stExpander {{
    background-color: var(--secondary-background-color);
    padding: 10px; /* Adiciona um pouco de padding */
    border-radius: 8px;
    margin-bottom: 15px; /* Espaço entre expanders */
}}

.stExpander > div > div > p {{
     color: var(--text-color); /* Garante que o texto dentro do expander seja visível */
}}

.stApp {{
    background-color: var(--background-color);
    color: var(--text-color); /* Garante que o texto geral do app use a cor definida */
}}

.stSidebar {{
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color);
}}

.stSidebar .stMarkdown {{
     color: var(--text-color); /* Garante que o markdown na sidebar use a cor definida */
}}

.stButton>button {{
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
    border: none; /* Remover borda padrão */
    padding: 10px 20px; /* Padding para melhor aparência */
    cursor: pointer;
}}

.stButton>button:hover {{
    background-color: #0099E6; /* Cor um pouco mais escura no hover */
}}

.stCheckbox>label {{
    color: var(--text-color);
}}

.stSelectbox>label {{
    color: var(--text-color);
}}

/* Melhorar aparência do date input */
.stDateInput > label {{
    color: var(--text-color);
}}

.stDateInput input {{
    color: var(--text-color);
    background-color: var(--secondary-background-color);
    border: 1px solid #555; /* Borda sutil */
    border-radius: 4px;
    padding: 5px;
}}

/* Melhorar aparência do slider */
.stSlider > label {{
    color: var(--text-color);
}}

.stSlider [data-baseweb="slider"] > div {{
    background-color: var(--primary-color); /* Cor da barra preenchida */
}}

.stSpinner > div > div {{
    color: var(--primary-color); /* Cor do spinner */
}}

/* Estilo para mensagens de aviso */
.stAlert > div {{
    background-color: rgba(255, 255, 0, 0.1); /* Amarelo semi-transparente */
    color: {TEXT_COLOR};
    border-color: yellow;
}}

/* Estilo para mensagens de erro */
.stAlert[kind="error"] > div {{
    background-color: rgba(255, 0, 0, 0.1); /* Vermelho semi-transparente */
    color: {TEXT_COLOR};
    border-color: red;
}}

/* Estilo para mensagens de sucesso */
.stAlert[kind="success"] > div {{
    background-color: rgba(0, 255, 0, 0.1); /* Verde semi-transparente */
    color: {TEXT_COLOR};
    border-color: green;
}}

/* Novos estilos para componentes de alertas */
.metric-container {{
    background-color: var(--secondary-background-color);
    border-radius: 8px;
    padding: 15px;
    margin: 5px;
}}

.metric-title {{
    color: var(--accent-color);
    font-size: 0.9rem;
}}

.metric-value {{
    color: var(--primary-color);
    font-size: 1.5rem;
    font-weight: bold;
}}


</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

# --- Funções de Banco de Dados e Carga ---

# Use st.secrets para credenciais de banco de dados
# Para configurar: crie um arquivo .streamlit/secrets.toml na raiz do seu projeto
# Exemplo:
# [mysql]
# host = "185.213.81.52"
# user = "u335174317_wazeportal"
# password = "@Ndre2025." # Mude isso para sua senha real ou use secrets
# database = "u335174317_wazeportal"

# faz conexxão com o banco de dados MySQL (cached)
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
            # Para a data de início, <= é correto se a hora for 00:00:00,
            # ou >= se quisermos incluir o início do dia. >= é mais comum.
            conditions.append("hr.data >= %s")
            params.append(start_date)
        if end_date:
            # CORREÇÃO: Para incluir o último dia completo, filtrar por < (data final + 1 dia)
            # Converte a string de data final para objeto datetime, adiciona 1 dia e converte de volta para string YYYY-MM-DD
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            end_date_plus_one_day_str = end_datetime.strftime('%Y-%m-%d') # Formatar como YYYY-MM-DD

            conditions.append("hr.data < %s") # Usar o operador MENOR QUE (<)
            params.append(end_date_plus_one_day_str) # Usar a data final + 1 dia

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY hr.data ASC"

        mycursor.execute(query, params)
        results = mycursor.fetchall()
        col_names = [desc[0] for desc in mycursor.description]
        df = pd.DataFrame(results, columns=col_names)

        # Convertendo 'data' para datetime e 'velocidade' para numérico
        df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None) # Remover timezone se presente
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')

        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e) # Retorna DataFrame vazio e erro
    finally:
        if mycursor:
            mycursor.close()


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

# --- Funções de Processamento e Análise ---

def clean_data(df):
    # Assume que o DataFrame já está filtrado pela rota e período
    df = df.copy()
    # df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None) # Já feito em get_data
    df = df.sort_values('data')
    df['velocidade'] = (
        df['velocidade']
        .clip(upper=150) # Limita a velocidade a 150 km/h
        .interpolate(method='linear') # Interpola valores ausentes linearmente
        .ffill() # Preenche valores restantes com o último valor válido
        .bfill() # Preenche valores restantes com o próximo valor válido
    )
    # Recalcular dia da semana e hora após interpolação/limpeza, se necessário
    df['day_of_week'] = df['data'].dt.day_name()
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade']) # Remove linhas onde a velocidade ainda é NaN

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


# Função de decomposição sazonal (revisada para usar índice de tempo e frequência)
def seasonal_decomposition_plot(df):
    if df.empty:
        st.info("Não há dados para realizar a decomposição sazonal.")
        return

    # Garantir frequência temporal, interpolando se houver lacunas curtas
    # Usa a coluna 'data' como índice e define a frequência como 3 minutos
    df_ts = df.set_index('data')['velocidade'].asfreq('3min')

    # Interpolar apenas se houver dados suficientes após asfreq
    if len(df_ts.dropna()) < len(df_ts) * 0.8: # Exemplo: exige pelo menos 80% dos dados para interpolar
         st.warning("Muitos dados faltantes para interpolação e decomposição sazonal confiáveis.")
         return

    df_ts = df_ts.interpolate(method='time')

    # O período para sazonalidade diária em dados de 3 em 3 minutos é 480 (24 horas * 60 min / 3 min)
    period = 480

    if len(df_ts.dropna()) < 2 * period:
         st.warning(f"Dados insuficientes para decomposição sazonal com período de {period}. Necessário pelo menos {2*period} pontos de dados válidos.")
         return

    try:
        # model='additive' é geralmente adequado para velocidade onde as variações são mais constantes
        decomposition = seasonal_decompose(df_ts.dropna(), model='additive', period=period)
        fig, ax = plt.subplots(4, 1, figsize=(12, 10)) # Adiciona componente de Resíduo
        decomposition.observed.plot(ax=ax[0], title='Observado')
        decomposition.trend.plot(ax=ax[1], title='Tendência')
        decomposition.seasonal.plot(ax=ax[2], title=f'Sazonalidade (Periodo {period})')
        decomposition.resid.plot(ax=ax[3], title='Resíduo')
        plt.tight_layout()

        # Configurar cores dos eixos e títulos para o tema escuro
        for a in ax:
            a.tick_params(axis='x', colors=TEXT_COLOR)
            a.tick_params(axis='y', colors=TEXT_COLOR)
            a.title.set_color(TEXT_COLOR)
            a.xaxis.label.set_color(TEXT_COLOR)
            a.yaxis.label.set_color(TEXT_COLOR)
            # O fundo dos subplots pode ser ajustado se necessário, mas tight_layout ajuda
            # a.set_facecolor(SECONDARY_BACKGROUND_COLOR)

        # Configurar cor de fundo da figura
        fig.patch.set_facecolor(SECONDARY_BACKGROUND_COLOR)

        st.pyplot(fig)
    except Exception as e:
         st.warning(f"Não foi possível realizar a decomposição sazonal: {e}")
         st.info("Verifique se os dados têm uma frequência regular ou se há dados suficientes.")


# Função de previsão ARIMA (revisada para usar intervalos de confiança e tratamento de dados)
def create_arima_forecast(df, route_id, steps=10):
    if df.empty:
        return pd.DataFrame()

    # Preparar dados para auto_arima (já vem limpo)
    # Garantir frequência temporal, interpolando se houver lacunas curtas
    arima_data = df.set_index('data')['velocidade'].asfreq('3min').dropna()

    if len(arima_data) < 24 * 7 * (60/3): # Exemplo: exige pelo menos 1 semana de dados com frequência de 3min para um ARIMA sazonal robusto
         st.warning("Dados insuficientes para um modelo de previsão ARIMA sazonal robusto. Necessário mais dados históricos (ex: pelo menos 1 semana).")
         return pd.DataFrame()

    try:
        # auto_arima encontrará os melhores parâmetros p,d,q,P,D,Q
        # m=480 para sazonalidade diária em dados de 3 em 3 minutos
        # Adicionado stepwise=True para acelerar, n_fits para limitar tentativas, random_state para reprodutibilidade
        model = auto_arima(arima_data, seasonal=True, m=480,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True, random_state=42,
                           n_fits=20) # Limitar o número de fits para evitar tempo excessivo

        # Realizar a previsão com intervalos de confiança
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)

        last_date = arima_data.index.max()
        # Gerar datas futuras com base na última data e frequência de 3 minutos
        # O start deve ser a última data observada, e periods = steps + 1 para incluir a primeira data da previsão
        # que vem APÓS a última data observada. Depois pegamos a partir do índice 1.
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
        st.warning("Não há previsão para salvar no banco de dados.")
        return # Não salva se o DataFrame estiver vazio

    # Ajustar nomes de colunas para corresponder à tabela forecast_history, se necessário
    # Assumindo que a tabela forecast_history tem colunas como 'data', 'previsao', 'limite_inferior', 'limite_superior', 'id_rota'
    # Ajuste conforme a estrutura real da sua tabela 'forecast_history'
    forecast_df_mapped = forecast_df.rename(columns={
        'ds': 'data',
        'yhat': 'previsao',
        'yhat_lower': 'limite_inferior',
        'yhat_upper': 'limite_superior',
        'id_route': 'id_rota'
    })

    try:
        # Usando credenciais do secrets
        engine = create_engine(
            f'mysql+mysqlconnector://{st.secrets["mysql"]["user"]}:{st.secrets["mysql"]["password"]}@{st.secrets["mysql"]["host"]}/{st.secrets["mysql"]["database"]}'
        )
        # Usando o gerenciador de contexto do SQLAlchemy para garantir commit/rollback e fechar a conexão
        with engine.begin() as connection:
             forecast_df_mapped.to_sql('forecast_history', con=connection, if_exists='append', index=False)
             st.success("Previsão salva no banco de dados!") # Feedback ao usuário
    except Exception as e:
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")

# --- Novas funções para alertas ---
def get_alerts(start_date=None, end_date=None, route_coords=None, max_distance_km=0.5):
    try:
        conn = get_db_connection()
        query = """
            SELECT 
                uuid, type, subtype, 
                location_x as longitude,
                location_y as latitude,
                pubMillis as data,
                reportRating as severidade,
                confidence as confianca,
                reliability as confiabilidade,
                street
            FROM alerts
            WHERE 1=1
        """
        conditions = []
        params = []

        # Filtro temporal corrigido
        if start_date:
            start_ts = int(pd.to_datetime(start_date).timestamp()) * 1000  # Parêntese fechado
            conditions.append("pubMillis >= %s")
            params.append(start_ts)
            
        if end_date:
            end_date_plus_1 = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            end_ts = int(end_date_plus_1.timestamp()) * 1000  # Sintaxe corrigida
            conditions.append("pubMillis < %s")
            params.append(end_ts)

        if conditions:
            query += " AND " + " AND ".join(conditions)

        df = pd.read_sql(query, conn, params=params)
        df['data'] = pd.to_datetime(df['data'], unit='ms')
        
        # Filtro espacial (mantido igual)
        if route_coords is not None and not route_coords.empty:
            from geopy.distance import great_circle
            route_points = list(zip(route_coords['latitude'], route_coords['longitude']))
            
            def is_near(point):
                return any(great_circle(point, route_point).km <= max_distance_km 
                          for route_point in route_points)
            
            df['near_route'] = df.apply(lambda row: is_near((row['latitude'], row['longitude'])), axis=1)
            df = df[df['near_route']]
        
        return df

    except Exception as e:
        st.error(f"Erro ao carregar alertas: {e}")
        return pd.DataFrame()

def check_alerts(data):
    alerts = []
    for rule_name, rule in ALERT_RULES.items():
        if rule['condition'](data).iloc[-1]:
            alerts.append(rule['message'].format(data['route_name'].iloc[0]))
    return alerts

def simulate_scenario(base_data, parameters):
    simulated = base_data.copy()
    # Aplicar parâmetros de simulação
    simulated['velocidade'] *= parameters.get('speed_factor', 1)
    # Adicionar ruído
    simulated['velocidade'] += np.random.normal(0, parameters.get('noise', 0))
    return simulated

def gerar_insights(df):
    insights = []
    if df.empty:
        return "Não há dados para gerar insights neste período."

    media_geral = df['velocidade'].mean()
    insights.append(f"📌 Velocidade média geral: **{media_geral:.2f} km/h**")

    # Encontrar o dia e hora mais lentos dentro do período selecionado
    if 'data' in df.columns and not df['data'].empty:
        daily_avg = df.groupby(df['data'].dt.date)['velocidade'].mean()
        if not daily_avg.empty:
            dia_mais_lento_date = daily_avg.idxmin()
            velocidade_dia_mais_lento = daily_avg.min()
            insights.append(f"📅 Dia com a menor velocidade média: **{dia_mais_lento_date}** ({velocidade_dia_mais_lento:.2f} km/h)")
        else:
             insights.append("Não foi possível calcular a velocidade média diária.")
    else:
         insights.append("Coluna 'data' não encontrada ou vazia no DataFrame para insights diários.")


    if 'day_of_week' in df.columns and not df['day_of_week'].empty:
        weekday_avg = df.groupby('day_of_week')['velocidade'].mean()
        if not weekday_avg.empty:
            # Reordenar para encontrar o dia da semana mais lento na ordem correta
            dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_avg = weekday_avg.reindex(dias_ordenados)
            dia_da_semana_mais_lento = weekday_avg.idxmin()
            insights.append(f"📅 Dia da semana mais lento (em média): **{dia_da_semana_mais_lento}**")
        else:
            insights.append("Não foi possível calcular a velocidade média por dia da semana.")
    else:
        insights.append("Coluna 'day_of_week' não encontrada ou vazia no DataFrame para insights por dia da semana.")

    if 'hour' in df.columns and not df['hour'].empty:
        hourly_avg = df.groupby('hour')['velocidade'].mean()
        if not hourly_avg.empty:
            hora_mais_lenta = hourly_avg.idxmin()
            insights.append(f"🕒 Hora do dia mais lenta (em média): **{hora_mais_lenta:02d}:00**")
        else:
             insights.append("Não foi possível calcular a velocidade média por hora do dia.")
    else:
         insights.append("Coluna 'hour' não encontrada ou vazia no DataFrame para insights por hora.")


    return "\n\n".join(insights)


# --- Função Principal do Aplicativo Streamlit ---

def main():
    # Verificar se as secrets do banco de dados estão configuradas
    if "mysql" not in st.secrets or not all(k in st.secrets["mysql"] for k in ("host", "user", "password", "database")):
        st.error("As credenciais do banco de dados não foram configuradas corretamente.")
        st.stop()

    # Inicialização de variáveis importantes
    routes_to_process = []
    routes_info = {}

    try:
        with st.sidebar:
            st.title("ℹ️ Painel de Controle")
            st.markdown("Configure a análise de rotas aqui.")

            # Carregar nomes das rotas
            all_route_names = get_all_route_names()
            if not all_route_names:
                st.error("Nenhuma rota encontrada no banco de dados!")
                st.stop()

            # Seleção de rotas
            main_route = st.selectbox(
                "Rota Principal:",
                all_route_names,
                index=0,
                key="main_route_select"
            )

            # Configuração de comparação
            compare_enabled = st.checkbox("Comparar com outra rota", key="compare_checkbox")
            second_route = None
            if compare_enabled:
                available_routes = [r for r in all_route_names if r != main_route]
                if available_routes:
                    second_route = st.selectbox(
                        "Rota Secundária:",
                        available_routes,
                        index=0,
                        key="secondary_route_select"
                    )
                else:
                    st.warning("Nenhuma outra rota disponível para comparação")
                    compare_enabled = False

            # Seleção de período
            st.subheader("Período de Análise")
            date_range = st.date_input(
                "Selecione o período:",
                value=[pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')],
                key="date_range"
            )

            # Validar datas
            if len(date_range) != 2:
                st.error("Selecione um intervalo de datas válido")
                st.stop()
                
            if date_range[0] > date_range[1]:
                st.error("A data final não pode ser anterior à data inicial")
                st.stop()

        # Configurar rotas para processamento
        routes_to_process = [main_route]
        if compare_enabled and second_route:
            routes_to_process.append(second_route)

        # --- Carregamento e Processamento de Dados ---
        st.title("🚀 Análise de Rotas Inteligente")
        st.header("⏳ Processando Dados...")
        
        for route in routes_to_process:
            with st.spinner(f'Carregando dados para {route}...'):
                raw_df, error = get_data(
                    start_date=date_range[0],
                    end_date=date_range[1],
                    route_name=route
                )

                if error:
                    st.error(f"Erro na rota {route}: {error}")
                    continue

                if raw_df.empty:
                    st.warning(f"Nenhum dado encontrado para {route}")
                    continue

                # Processar dados
                route_id = raw_df['route_id'].iloc[0]
                processed_df = clean_data(raw_df)
                routes_info[route] = {
                    'data': processed_df,
                    'id': route_id,
                    'coords': get_route_coordinates(route_id)
                }

        # Verificar se há dados válidos
        if not routes_info:
            st.error("Nenhum dado válido encontrado para análise")
            st.stop()

        # --- Seção de Visualização Geográfica ---
        st.header("🗺️ Visualização Geográfica")
        for route, data in routes_info.items():
            with st.expander(f"Mapa: {route}", expanded=True):
                # Criar mapa
                fig = go.Figure()
                
                # Adicionar rota
                if not data['coords'].empty:
                    fig.add_trace(go.Scattermapbox(
                        mode="lines",
                        lon=data['coords']['longitude'],
                        lat=data['coords']['latitude'],
                        line=dict(width=4, color=PRIMARY_COLOR),
                        name='Rota'
                    ))

                # Carregar e plotar alertas
                alerts_df = get_alerts(
                    start_date=date_range[0],
                    end_date=date_range[1],
                    route_coords=data['coords']
                )
                
                if not alerts_df.empty:
                    fig.add_trace(go.Scattermapbox(
                        mode="markers",
                        lon=alerts_df['longitude'],
                        lat=alerts_df['latitude'],
                        marker=dict(size=10, color=ACCENT_COLOR),
                        text=alerts_df['type'] + " - " + alerts_df['subtype'],
                        name='Alertas'
                    ))

                # Configurar layout do mapa
                fig.update_layout(
                    mapbox_style="carto-darkmatter",
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Seção de Análise Preditiva ---
        st.header("📈 Análise de Desempenho")
        for route, data in routes_info.items():
            with st.expander(f"Análise para {route}", expanded=True):
                # Insights e visualizações
                st.subheader("🧠 Insights Automáticos")
                st.markdown(gerar_insights(data['data']))
                
                # Previsão ARIMA
                st.subheader("🔮 Previsão de Velocidade")
                steps = st.slider(f"Horizonte de previsão (minutos)", 15, 120, 60, key=f"steps_{route}")
                
                if st.button(f"Gerar Previsão para {route}"):
                    forecast = create_arima_forecast(data['data'], data['id'], steps)
                    if not forecast.empty:
                        # Plotar gráfico de previsão
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data['data']['data'],
                            y=data['data']['velocidade'],
                            name='Histórico'
                        ))
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'],
                            y=forecast['yhat'],
                            name='Previsão',
                            line=dict(dash='dot')
                        ))
                        st.plotly_chart(fig)

        # --- Seção de Alertas ---
        st.header("🚨 Análise de Alertas")
        if not alerts_df.empty:
            with st.expander("Detalhes dos Alertas", expanded=True):
                # Filtros interativos
                col1, col2 = st.columns(2)
                selected_type = col1.selectbox("Tipo de Alerta", ['Todos'] + list(alerts_df['type'].unique()))
                min_severity = col2.slider("Severidade Mínima", 0, 5, 0)

                # Aplicar filtros
                filtered_alerts = alerts_df.copy()
                if selected_type != 'Todos':
                    filtered_alerts = filtered_alerts[filtered_alerts['type'] == selected_type]
                filtered_alerts = filtered_alerts[filtered_alerts['severidade'] >= min_severity]

                # Exibir resultados
                st.dataframe(filtered_alerts)
        else:
            st.info("Nenhum alerta encontrado no período selecionado")

        # --- Seção Técnica ---
        st.header("⚙️ Detalhes Técnicos")
        with st.expander("Relatório de Qualidade de Dados"):
            for route, data in routes_info.items():
                st.subheader(f"Relatório para {route}")
                st.json({
                    "registros_processados": len(data['data']),
                    "periodo_cobertura": f"{data['data']['data'].min()} a {data['data']['data'].max()}",
                    "alertas_associados": len(alerts_df[alerts_df['route_name'] == route])
                })

    except Exception as e:
        st.error(f"Erro crítico na aplicação: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()