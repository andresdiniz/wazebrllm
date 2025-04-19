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
        st.error("As credenciais do banco de dados não foram configuradas corretamente no secrets.toml.")
        st.markdown("Por favor, crie ou atualize o arquivo `.streamlit/secrets.toml` com as informações de conexão do MySQL.")
        st.stop() # Parar a execução

    with st.sidebar:
        st.title("ℹ️ Painel de Controle")
        st.markdown("""
            Configure a análise de rotas aqui.

            **Funcionalidades:**
            - Visualize dados históricos de velocidade
            - Detecte padrões de tráfego (heatmap, decomposição)
            - Obtenha insights automáticos sobre a rota
            - Previsão de velocidade para o futuro próximo
            - Compare a análise entre diferentes rotas
        """)

        st.subheader("Seleção de Rotas")
        # Carregar nomes das rotas de forma eficiente (cached)
        all_route_names = get_all_route_names()
        if not all_route_names:
             st.warning("Não foi possível carregar os nomes das rotas do banco de dados ou não há rotas disponíveis.")
             st.stop() # Parar se não houver rotas

        # Usar índice para garantir que o selectbox não quebre se o nome da rota mudar ou não existir
        try:
            default_main_route_index = all_route_names.index(st.session_state.get("main_route_select", all_route_names[0]))
        except ValueError:
             default_main_route_index = 0 # Usar o primeiro se o valor armazenado não for válido

        route_name = st.selectbox(
            "Rota Principal:",
            all_route_names,
            index=default_main_route_index,
            key="main_route_select"
        )

        compare_enabled = st.checkbox("Comparar com outra rota", key="compare_checkbox")
        second_route = None
        if compare_enabled:
            available_for_comparison = [r for r in all_route_names if r != route_name]
            if available_for_comparison:
                 # Usar índice para a rota secundária também
                 try:
                     default_secondary_route_index = available_for_comparison.index(st.session_state.get("secondary_route_select", available_for_comparison[0]))
                 except ValueError:
                      default_secondary_route_index = 0

                 second_route = st.selectbox(
                     "Rota Secundária:",
                     available_for_comparison,
                     index=default_secondary_route_index,
                     key="secondary_route_select"
                 )
            else:
                 st.info("Não há outras rotas disponíveis para comparação.")
                 compare_enabled = False # Desabilita comparação se não houver outras rotas


        st.subheader("Período de Análise")
        # Usar um seletor de data por rota para flexibilidade na comparação de períodos diferentes
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_range_main = st.date_input(
                f"Período para '{route_name}'",
                value=((pd.to_datetime('today') - pd.Timedelta(days=7)).date(), pd.to_datetime('today').date()), # CORRIGIDO AQUI
                max_value=pd.to_datetime('today').date(),
                key=f"date_range_{route_name}"
            )

        date_range_secondary = None
        if compare_enabled and second_route:
             with col_date2:
                 date_range_secondary = st.date_input(
                    f"Período para '{second_route}'",
                    value=((pd.to_datetime('today') - pd.Timedelta(days=7)).date(), pd.to_datetime('today').date()), # CORRIGIDO AQUI
                    max_value=pd.to_datetime('today').date(),
                    key=f"date_range_{second_route}"
                )
                 # A validação de data final anterior à inicial já está logo abaixo, isso é bom
                 # if date_range_secondary[0] > date_range_secondary[1]:
                 #     st.error("Data final da rota secundária não pode ser anterior à data inicial.")
                 #     st.stop()


        # Validar as datas (este bloco já estava correto)
        if date_range_main and date_range_main[0] > date_range_main[1]:
            st.error("Data final da rota principal não pode ser anterior à data inicial")
            st.stop()
        if compare_enabled and date_range_secondary and date_range_secondary[0] > date_range_secondary[1]:
             st.error("Data final da rota secundária não pode ser anterior à data inicial.")
             st.stop()

    st.title("🚀 Análise de Rotas Inteligente")
    st.markdown("Selecione as rotas e o período de análise no painel lateral.")

    routes_info = {}
    routes_to_process = [route_name]
    if compare_enabled and second_route:
        routes_to_process.append(second_route)

    # --- Carregamento e Processamento de Dados ---
    st.header("⏳ Processando Dados...")
    for route in routes_to_process:
        date_range = date_range_main if route == route_name else date_range_secondary
        if date_range is None: # Caso a comparação esteja habilitada, mas a rota secundária não tenha range
             continue

        start_date = date_range[0].strftime('%Y-%m-%d')
        end_date = date_range[1].strftime('%Y-%m-%d')

        with st.spinner(f'Carregando e processando dados para {route} de {start_date} a {end_date}...'):
            # Carregar dados filtrando por nome da rota e período (cached)
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
                st.warning(f"Nenhum dado encontrado para a rota '{route}' no período de {start_date} a {end_date}. Por favor, ajuste o intervalo de datas.")
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
        st.success(f"Dados para {route} carregados e processados ({len(processed_df)} registros).")

    # --- Seção de Visualização ---

    # Se não houver dados carregados para nenhuma rota, parar por aqui
    if not routes_info or all(info['data'].empty for info in routes_info.values()):
         st.info("Selecione as rotas e um período com dados disponíveis no painel lateral para continuar.")
         return # Sai da função main se não houver dados


    st.header("🗺️ Visualização Geográfica")
    for route in routes_to_process:
        if route in routes_info and not routes_info[route]['data'].empty:
            route_id = routes_info[route]['id']
            with st.expander(f"Mapa da Rota: {route}", expanded=True):
                # Obter coordenadas da rota (cached)
                route_coords = get_route_coordinates(route_id)

                if not route_coords.empty:
                    # Calcular bounds para centralizar o mapa
                    min_lat, max_lat = route_coords['latitude'].min(), route_coords['latitude'].max()
                    min_lon, max_lon = route_coords['longitude'].min(), route_coords['longitude'].max()

                    # Adicionar um pequeno buffer
                    lat_buffer = (max_lat - min_lat) * 0.05
                    lon_buffer = (max_lon - min_lon) * 0.05

                    center_lat = (max_lat + min_lat) / 2
                    center_lon = (max_lon + min_lon) / 2

                    # Determinar um zoom inicial razoável baseado nos bounds
                    # Pode ser necessário ajustar esta lógica dependendo da escala das suas rotas
                    zoom = 12 # Valor padrão

                    fig = go.Figure(go.Scattermapbox(
                        mode="lines+markers",
                        lon=route_coords['longitude'],
                        lat=route_coords['latitude'],
                        # CORRIGIDO: Use o valor hexadecimal da variável --accent-color
                        marker={'size': 8, 'color': ACCENT_COLOR},
                        # CORRIGIDO: Use o valor hexadecimal da variável --primary-color
                        line=dict(width=4, color=PRIMARY_COLOR),
                        hovertext=[f"Ponto {i+1}" for i in range(len(route_coords))],
                        hoverinfo="text+lat+lon"
                    ))

                    fig.update_layout(
                        mapbox={
                            'style': "carto-darkmatter", # Estilo de mapa que combina com o tema escuro
                            'center': {'lat': center_lat, 'lon': center_lon},
                            'zoom': zoom,
                             # Bounds podem ajudar a focar na área, mas 'center' e 'zoom' são mais comuns
                             'bounds': {'west': min_lon - lon_buffer, 'east': max_lon + lon_buffer,
                                        'south': min_lat - lat_buffer, 'north': max_lat + lat_buffer}
                        },
                        margin={"r":0,"t":0,"l":0,"b":0},
                        height=500, # Altura do mapa
                        # CORRIGIDO: Use o valor hexadecimal da variável --secondary-background-color
                        plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                        # CORRIGIDO: Use o valor hexadecimal da variável --secondary-background-color
                        paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                        # CORRIGIDO: Use o valor hexadecimal da variável --text-color
                        font=dict(color=TEXT_COLOR)
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
                    # CORRIGIDO: Usar a cor do tema
                    ax.set_title("Velocidade Média por Dia da Semana e Hora", color=TEXT_COLOR)
                    # CORRIGIDO: Usar a cor do tema
                    ax.set_xlabel("Hora do Dia", color=TEXT_COLOR)
                    # CORRIGIDO: Usar a cor do tema
                    ax.set_ylabel("Dia da Semana", color=TEXT_COLOR)
                    # Ajustar cor dos ticks e labels
                    # CORRIGIDO: Usar a cor do tema
                    ax.tick_params(axis='x', colors=TEXT_COLOR)
                    # CORRIGIDO: Usar a cor do tema
                    ax.tick_params(axis='y', colors=TEXT_COLOR)
                    # Mudar cor do background do plot
                    # CORRIGIDO: Usar a cor do tema
                    fig.patch.set_facecolor(SECONDARY_BACKGROUND_COLOR)
                    # CORRIGIDO: Usar a cor do tema
                    ax.set_facecolor(SECONDARY_BACKGROUND_COLOR)
                    st.pyplot(fig)
                else:
                     st.info("Dados insuficientes para gerar o Heatmap.")


                st.subheader("🔮 Previsão de Velocidade (Modelo ARIMA)")
                # Certificar que o DataFrame tem dados e foi limpo
                if not processed_df.empty:
                     # A frequência da série temporal é 3 minutos, baseada na coleta
                     # O key do slider precisa ser único por rota
                     steps = st.slider(f"⏱️ Passos de previsão para '{route}' (3min cada)", 5, 60, 10, key=f"steps_{route}")

                     # Criar previsão (cached)
                     arima_forecast = create_arima_forecast(processed_df, route_id, steps)

                     if not arima_forecast.empty:
                         fig = go.Figure()

                         # Adicionar dados históricos
                         fig.add_trace(go.Scatter(
                             x=processed_df['data'],
                             y=processed_df['velocidade'],
                             mode='lines',
                             name='Histórico',
                             # CORRIGIDO: Use o valor hexadecimal da variável --primary-color
                             line=dict(color=PRIMARY_COLOR, width=2)
                         ))

                         # Adicionar previsão
                         fig.add_trace(go.Scatter(
                             x=arima_forecast['ds'],
                             y=arima_forecast['yhat'],
                             mode='lines',
                             name='Previsão',
                             # CORRIGIDO: Use o valor hexadecimal da variável --accent-color
                             line=dict(color=ACCENT_COLOR, width=2, dash='dash')
                         ))

                         # Adicionar intervalo de confiança
                         # Usando a cor baseada no primary color com transparência
                         fill_color_rgba = f'rgba({int(PRIMARY_COLOR[1:3], 16)}, {int(PRIMARY_COLOR[3:5], 16)}, {int(PRIMARY_COLOR[5:7], 16)}, 0.2)'

                         fig.add_trace(go.Scatter(
                             x=arima_forecast['ds'].tolist() + arima_forecast['ds'][::-1].tolist(), # Datas para preencher a área
                             y=arima_forecast['yhat_upper'].tolist() + arima_forecast['yhat_lower'][::-1].tolist(), # Limites para preencher a área
                             fill='toself',
                             # CORRIGIDO: Use a cor RGBA baseada no tema
                             fillcolor=fill_color_rgba,
                             line=dict(color='rgba(255,255,255,0)'), # Linha transparente
                             name='Intervalo de Confiança (95%)',
                             showlegend=True
                         ))

                         fig.update_layout(
                             title=f"Histórico e Previsão para {route}",
                             xaxis_title="Data e Hora",
                             yaxis_title="Velocidade (km/h)",
                             hovermode='x unified', # Melhorar hover
                             # CORRIGIDO: Use o valor hexadecimal da variável --secondary-background-color
                             plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                             # CORRIGIDO: Use o valor hexadecimal da variável --secondary-background-color
                             paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                             # CORRIGIDO: Use o valor hexadecimal da variável --text-color
                             font=dict(color=TEXT_COLOR),
                             # Manter as cores do grid em cinza, combinam com o tema escuro
                             xaxis=dict(showgrid=True, gridcolor='#555'),
                             yaxis=dict(showgrid=True, gridcolor='#555'),
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Posicionar legenda
                         )
                         st.plotly_chart(fig)

                         # Botão para salvar a previsão (key precisa ser única por rota)
                         if st.button(f"💾 Salvar Última Previsão para '{route}'", key=f"save_forecast_{route}"):
                              save_forecast_to_db(arima_forecast)

                     else:
                         st.info("Não foi possível gerar a previsão para esta rota e período. Verifique se há dados históricos suficientes.")
                else:
                     st.info("Dados insuficientes para rodar a análise de previsão para esta rota e período.")

        elif route in routes_info and 'error' in routes_info[route]:
             st.warning(f"Análise preditiva não disponível para '{route}' devido a erro no carregamento de dados.")


    # --- Seção Técnica ---
    st.header("⚙️ Detalhes Técnicos")
    with st.expander("Relatório de Qualidade de Dados"):
        for route in routes_to_process:
            # Verificar se a rota foi processada e não teve erro de carga fatal
            if route in routes_info and 'error' not in routes_info[route]:
                st.subheader(f"Qualidade dos Dados: {route}")
                processed_df = routes_info[route]['data']

                if not processed_df.empty:
                    report = {
                        "total_registros_carregados_periodo": len(processed_df),
                        "registros_velocidade_nulos_apos_limpeza": processed_df['velocidade'].isnull().sum(), # Deve ser 0 se o ffill/bfill funcionou
                        # anomaly detection needs to be implemented and used if desired in the report
                        # "outliers_detectados": len(detect_anomalies(processed_df)), # Usar se a detecção de anomalias for usada
                        "cobertura_temporal": f"{processed_df['data'].min().strftime('%Y-%m-%d %H:%M')} a {processed_df['data'].max().strftime('%Y-%m-%d %H:%M')}" if not processed_df.empty else "N/A"
                    }
                    st.json(report)
                else:
                     st.info(f"Não há dados processados para gerar relatório de qualidade para '{route}'.")
            elif route in routes_info and 'error' in routes_info[route]:
                 st.warning(f"Relatório de qualidade não disponível para '{route}' devido a erro no carregamento de dados: {routes_info[route]['error']}")
            else:
                 st.info(f"Dados para '{route}' não foram carregados ou processados.")

                 # Exemplo de integração de alertas
                 ALERT_RULES = {
                    "congestion": {
                        "condition": lambda df: df['velocidade'].rolling(4).mean() < 20,
                        "duration": "15min",
                        "message": "Congestionamento formando na rota {}"
                    }
                }

if __name__ == "__main__":
    main()