import streamlit as st
import logging # Importar a biblioteca de logging

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
# Importar matplotlib e seaborn para o heatmap e decomposição
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px # Mantido para caso precise em outros lugares (mapa usa go, previsão usa go)
import plotly.graph_objects as go
from io import BytesIO
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import datetime # Importar datetime para manipular dates
import holidays # Importar a biblioteca holidays para feriados


# Configurações de compatibilidade do numpy (manter se for necessário no seu ambiente)
# Isso pode não ser necessário dependendo da versão do numpy, mas é seguro manter
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

# Tema personalizado MELHORADO E COERENTE COM FUNDO ESCURO
# Definindo as cores em variáveis para fácil referência
PRIMARY_COLOR = "#00AFFF"         # Azul mais claro e vibrante
BACKGROUND_COLOR = "#1E1E1E"      # Cinza escuro para o fundo principal
SECONDARY_BACKGROUND_COLOR = "#2D2D2D" # Cinza um pouco mais claro para sidebar/elementos
ACCENT_COLOR = "#FF4B4B"          # Vermelho para destaque/alertas
TEXT_COLOR = "#FFFFFF"            # Branco
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
/* Ajustar cor do header do expander */
.stExpander > div > div > .st-emotion-cache-p5msec {{
    color: var(--text-color); /* Garante que o título do expander seja visível */
}}


.stApp {{
    background-color: var(--background-color);
    color: var(--text-color); /* Garante que o texto geral do app use a cor definida */
}}

/* --- Ajustes específicos para a sidebar (MAIS AGRESSIVOS) --- */
/* Usando 'background' shorthand e '!important' mais assertivamente */
.stSidebar {{
    background: var(--secondary-background-color) !important; /* Força o fundo escuro */
    color: var(--text-color) !important; /* Força a cor do texto geral */
    /* Adicionar propriedades para garantir que cubra a área corretamente, se necessário */
    /* height: 100vh !important; */
    /* position: fixed !important; width: 210px !important; top: 0; left: 0; */
}}

.stSidebar .stMarkdown {{
     color: var(--text-color) !important; /* Força cor para markdown */
}}

/* Forçar a cor do texto para elementos de input e labels dentro da sidebar */
.stSidebar label {{
    color: var(--text-color) !important;
}}

.stSidebar div[data-baseweb="select"] > div {{
     background-color: var(--secondary-background-color) !important;
     color: var(--text-color) !important;
     border: 1px solid #555;
}}

.stSidebar input[type="text"],
.stSidebar input[type="date"],
.stSidebar input[type="number"]
{{
    color: var(--text-color) !important;
    background-color: var(--secondary-background-color) !important;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
}}

.stSidebar .stSlider [data-baseweb="slider"] > div {{
    background-color: var(--primary-color) !important;
}}

.stSidebar .stRadio > label {{
     color: var(--text-color) !important;
}}

/* Garantir que o texto dos botões na sidebar seja visível */
.stSidebar button {{
    color: white !important; /* Força a cor do texto do botão para branco */
}}

/* --- Fim ajustes sidebar --- */


.stButton>button {{
    background-color: var(--primary-color);
    color: white;
    border-radius: 8后mpx;
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
/* Melhorar aparência do selectbox - Regra global */
.stSelectbox > div[data-baseweb="select"] > div {{
     background-color: var(--secondary-background-color);
     color: var(--text-color);
     border: 1px solid #555;
}}


/* Melhorar aparência do date input - Regra global */
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

/* Melhorar aparência do slider - Regra global */
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

/* Adiciona hover effect nos botões */
.stButton button:hover {{
    opacity: 0.9;
    transform: scale(1.02);
    transition: all 0.2s ease-in-out;
}}
/* Ajustar o padding da página principal */
.stApp > header, .stApp > div {{
    padding-top: 1rem;
    padding-bottom: 1rem;
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

def get_db_connection():
    """
    Estabelece e retorna uma conexão com o banco de dados MySQL.
    A conexão é cacheada pelo Streamlit.
    """
    try:
        # Configuração de pooling ou outras otimizações podem ser adicionadas aqui
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except Exception as e:
        logging.exception("Erro ao conectar ao banco de dados:") # Log detalhado
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.stop() # Parar a execução se não conseguir conectar

def get_data(start_date=None, end_date=None, route_name=None):
    """
    Busca dados históricos de velocidade do banco de dados para uma rota e período específicos.

    Args:
        start_date (str, optional): Data de início no formato YYYY-MM-DD. Defaults to None.
        end_date (str, optional): Data de fim no formato YYYY-MM-DD. Defaults to None.
        route_name (str, optional): Nome da rota. Defaults to None.

    Returns:
        tuple[pd.DataFrame, str | None]: DataFrame com os dados e mensagem de erro (se houver).
    """
    mydb = None
    mycursor = None
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()

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
             # Para incluir o último dia completo, filtrar por < (data final + 1 dia)
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            end_date_plus_one_day_str = end_datetime.strftime('%Y-%m-%d')

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
        logging.exception(f"Erro ao obter dados para rota {route_name}:") # Log detalhado
        return pd.DataFrame(), str(e) # Retorna DataFrame vazio e erro
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

def get_all_route_names():
    """
    Busca todos os nomes de rotas distintos no banco de dados.
    A lista de nomes é cacheada pelo Streamlit.

    Returns:
        list[str]: Lista de nomes de rotas.
    """
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
        logging.exception("Erro ao obter nomes das rotas:") # Log detalhado
        st.error(f"Erro ao obter nomes das rotas: {e}")
        return []
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

def get_route_coordinates(route_id):
    """
    Busca as coordenadas geográficas (linha) para uma rota específica.
    As coordenadas são cacheadas pelo Streamlit.

    Args:
        route_id (int): ID da rota.

    Returns:
        pd.DataFrame: DataFrame com colunas 'longitude' e 'latitude'.
    """
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
        logging.exception(f"Erro ao obter coordenadas para route_id {route_id}:") # Log detalhado
        st.error(f"Erro ao obter coordenadas: {e}")
        return pd.DataFrame()
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

# --- Funções de Processamento e Análise ---

def clean_data(df):
    """
    Limpa, interpola e adiciona features temporais a um DataFrame de velocidade.

    Args:
        df (pd.DataFrame): DataFrame bruto com colunas 'data' e 'velocidade'.

    Returns:
        pd.DataFrame: DataFrame limpo com 'day_of_week' e 'hour' adicionados.
                      Retorna DataFrame vazio se todas as velocidades forem nulas.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Adicionar validação de dados: verificar se todas as velocidades estão nulas
    if df['velocidade'].isnull().all():
        st.warning("Após o carregamento, todas as velocidades estão nulas. Verifique os dados de origem ou o período selecionado.")
        return pd.DataFrame() # Retorna DataFrame vazio se todos os valores são nulos

    # Assume que o DataFrame já está filtrado pela rota e período
    # e que a coluna 'data' já é datetime sem timezone e 'velocidade' é numérica
    df = df.sort_values('data')
    df['velocidade'] = (
        df['velocidade']
        .clip(upper=150) # Limita a velocidade a 150 km/h
        .interpolate(method='linear') # Interpola valores ausentes linearmente
        .ffill() # Preenche valores restantes com o último valor válido
        .bfill() # Preenche valores restantes com o próximo valor válido
    )
    # Recalcular dia da semana e hora após interpolação/limpeza, se necessário
    # Usar locale para nomes dos dias em português
    # import locale
    # locale.setlocale(locale.LC_TIME, 'pt_BR.UTF8') # Configurar localidade (pode precisar instalar no ambiente)
    df['day_of_week'] = df['data'].dt.day_name() # Retorna em inglês por padrão, mapearemos para o heatmap
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade']) # Remove linhas onde a velocidade ainda é NaN


def seasonal_decomposition_plot(df):
    """
    Realiza e plota a decomposição sazonal de uma série temporal de velocidade.

    Args:
        df (pd.DataFrame): DataFrame com dados limpos e índice de tempo.
    """
    if df.empty:
        st.info("Não há dados para realizar a decomposição sazonal.")
        return

    # Garantir frequência temporal, interpolando se houver lacunas curtas
    # Usa a coluna 'data' como índice e define a frequência como 3 minutos
    df_ts = df.set_index('data')['velocidade'].asfreq('3min')

    # Interpolar apenas se houver dados suficientes após asfreq
    # Verifica a proporção de NaNs antes de interpolar
    if df_ts.isnull().sum() / len(df_ts) > 0.2: # Exemplo: Se mais de 20% dos dados são NaN após asfreq
         st.warning("Muitos dados faltantes ou espaçados para interpolação e decomposição sazonal confiáveis.")
         return

    df_ts = df_ts.interpolate(method='time')

    # O período para sazonalidade diária em dados de 3 em 3 minutos é 480 (24 horas * 60 min / 3 min)
    period = 480 # Usando o período padrão

    # Precisa de pelo menos 2 ciclos completos de dados para decomposição sazonal
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
            # Fundo dos subplots
            a.set_facecolor(SECONDARY_BACKGROUND_COLOR)

        # Configurar cor de fundo da figura
        fig.patch.set_facecolor(SECONDARY_BACKGROUND_COLOR)

        st.pyplot(fig)
    except Exception as e:
         logging.exception("Erro ao realizar decomposição sazonal:") # Log detalhado
         st.warning(f"Não foi possível realizar a decomposição sazonal: {e}")
         st.info("Verifique se os dados têm uma frequência regular ou se há dados suficientes.")


def create_holiday_exog(index):
    """
    Cria features exógenas binárias ('is_holiday' e 'is_pre_holiday') para um DateTimeIndex.

    Args:
        index (pd.DateTimeIndex): Índice de tempo para o qual gerar as features.

    Returns:
        pd.DataFrame: DataFrame com as colunas 'is_holiday' e 'is_pre_holiday'.
    """
    if index.empty:
        return pd.DataFrame(index=index)

    # Obter feriados brasileiros para os anos presentes no índice
    br_holidays = holidays.CountryHoliday('BR', years=index.year.unique())
    exog_df = pd.DataFrame(index=index)

    # is_holiday: Verifica se a data do timestamp atual é um feriado
    exog_df['is_holiday'] = index.to_series().apply(lambda date: date.date() in br_holidays).astype(int)

    # is_pre_holiday: Verifica se a data EXATAMENTE 24 horas a partir do timestamp atual é um feriado,
    # E a data atual NÃO é um feriado.
    # Isso requer que o índice tenha uma frequência regular definida por asfreq.
    if index.freq is None:
         # Fallback para frequência irregular - verifica se o próximo dia CALENDAR (24h) é um feriado
         exog_df['is_pre_holiday'] = index.to_series().apply(
             lambda date: (date + pd.Timedelta(days=1)).date() in br_holidays and date.date() not in br_holidays
         ).astype(int)
    else:
        # Usa a frequência para calcular um offset exato de 24 horas
        one_day_offset = pd.Timedelta(days=1)
        # Cria uma série de dates exatamente 24 horas no futuro com base na frequência do índice
        dates_in_24h = index + one_day_offset
        # Verifica se a data 24 hours later é um feriado
        is_next_day_holiday = dates_in_24h.to_series().apply(lambda date: date.date() in br_holidays).astype(int)
        # Uma data é véspera de feriado se a data 24h later é feriado E a data atual NÃO é feriado
        exog_df['is_pre_holiday'] = is_next_day_holiday & (exog_df['is_holiday'] == 0)

    return exog_df


# Função de previsão ARIMA (revisada para usar intervalos de confiança e tratamento de dados E EXOG)
# Não cacheamos previsões pois elas dependem de dados recentes e podem ser acionadas pelo usuário
# @st.cache_data # Não use cache_data para previsões se elas devem ser geradas sob demanda
def create_arima_forecast(df, route_id, steps=10, m_period=480):
    """
    Cria e executa um modelo de previsão ARIMA sazonal com variáveis exógenas (feriados/vésperas).

    Args:
        df (pd.DataFrame): DataFrame com dados históricos de velocidade limpos.
        route_id (int): ID da rota.
        steps (int, optional): Número de passos futuros para prever. Defaults to 10.
        m_period (int, optional): Período sazonal para o auto_arima. Defaults to 480 (diário @ 3min).

    Returns:
        pd.DataFrame: DataFrame com a previsão (datas, yhat, limites de confiança) ou DataFrame vazio em caso de falha.
    """
    if df.empty:
        # Mensagem já exibida na chamada
        return pd.DataFrame()

    # Preparar dados para auto_arima (já vem limpo)
    # Garantir frequência temporal, interpolando se houver lacunas curtas
    arima_data_full = df.set_index('data')['velocidade'].asfreq('3min').dropna()

    # Criar features exógenas (feriados e vésperas) para o período dos dados históricos
    exog_data_full = create_holiday_exog(arima_data_full.index)

    # Alinhar dados da série temporal (y) e dados exógenos (X) usando um join interno
    # Isso garante que temos 'y' e 'X' para os mesmos timestamps
    combined_df = arima_data_full.to_frame(name='y').join(exog_data_full, how='inner').dropna()
    arima_data = combined_df['y']
    exog_data = combined_df[['is_holiday', 'is_pre_holiday']]


    # Precisa de dados suficientes para o modelo sazonal ARIMA
    # Um mínimo de 2-3 ciclos sazonais é recomendado
    min_data_points = 2 * m_period # Mínimo 2 ciclos completos para detectar sazonalidade

    if len(arima_data) < min_data_points:
         st.warning(f"Dados insuficientes ({len(arima_data)} pontos) para treinar um modelo de previsão ARIMA sazonal robusto com período {m_period}. Necessário pelo menos {int(min_data_points)} pontos válidos após alinhamento.")
         return pd.DataFrame()

    try:
        # auto_arima encontrará os melhores parâmetros p,d,q,P,D,Q
        # Passando o período sazonal 'm' selecionado pelo usuário
        # PASSANDO DADOS EXÓGENOS (X=exog_data)
        with st.spinner(f"Treinando modelo ARIMA para a rota {route_id} com período sazonal m={m_period}..."):
             model = auto_arima(arima_data, X=exog_data, seasonal=True, m=m_period,
                                error_action='ignore', suppress_warnings=True,
                                stepwise=True, random_state=42,
                                n_fits=20) # Limitar o número de fits para evitar tempo excessivo

        # Gerar dates futuras com base na última data histórica e frequência
        last_date = arima_data.index.max()
        # A frequência deve ser compatível com m=480 ou 3360 (baseado em 3min)
        freq_str = '3min' # Assumindo 3minutos como base

        future_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq_str)[1:]

        # Criar features exógenas (feriados e vésperas) para o PERÍODO DA PREVISÃO
        future_exog_data = create_holiday_exog(future_dates)
        # Garantir que o índice dos dados exógenos futuros corresponda exatamente às dates futuras
        future_exog_data = future_exog_data.reindex(future_dates)


        # Realizar a previsão com intervalos de confiança
        # PASSANDO DADOS EXÓGENOS FUTUROS (X=future_exog_data)
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True, X=future_exog_data)


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
        logging.exception("Erro durante o treinamento ou previsão do modelo ARIMA:") # Log detalhado
        st.error(f"Erro durante o treinamento ou previsão do modelo ARIMA: {str(e)}")
        st.info("Verifique os dados de entrada, a quantidade de dados, ou a configuração do modelo ARIMA.")
        return pd.DataFrame()


def save_forecast_to_db(forecast_df):
    """
    Salva um DataFrame de previsão no banco de dados.

    Args:
        forecast_df (pd.DataFrame): DataFrame com a previsão a ser salva.
    """
    if forecast_df.empty:
        st.warning("Não há previsão para salvar no banco de dados.")
        return # Não salva se o DataFrame estiver vazio

    # Ajustar nomes de colunas para corresponder à tabela forecast_history
    # Assumindo que a tabela forecast_history tem colunas como 'data', 'previsao', 'limite_inferior', 'limite_superior', 'id_rota'
    forecast_df_mapped = forecast_df.rename(columns={
        'ds': 'data',
        'yhat': 'previsao',
        'yhat_lower': 'limite_inferior',
        'yhat_upper': 'limite_superior',
        'id_route': 'id_rota'
    })

    # Selecionar apenas as colunas que você quer salvar
    cols_to_save = ['data', 'previsao', 'limite_inferior', 'limite_superior', 'id_rota']
    forecast_df_mapped = forecast_df_mapped[cols_to_save]

    try:
        # st.info("Conectando ao banco de dados para salvar previsão...") # Substituído por toast/log
        # Usando credenciais do secrets
        engine = create_engine(
            f'mysql+mysqlconnector://{st.secrets["mysql"]["user"]}:{st.secrets["mysql"]["password"]}@{st.secrets["mysql"]["host"]}/{st.secrets["mysql"]["database"]}'
        )
        # Usando o gerenciador de contexto do SQLAlchemy para garantir commit/rollback e fechar a conexão
        # if_exists='append' adiciona novas linhas. Se você precisar evitar duplicatas,
        # pode precisar de uma lógica de upsert ou verificar antes de inserir.
        with engine.begin() as connection:
             # st.info("Salvando previsão na tabela forecast_history...") # Substituído por toast/log
             # Converte datetime para tipo compatível com SQL, como string ou timestamp
             forecast_df_mapped['data'] = forecast_df_mapped['data'].dt.strftime('%Y-%m-%d %H:%M:%S')
             forecast_df_mapped.to_sql('forecast_history', con=connection, if_exists='append', index=False)
             st.toast("Previsão salva no banco de dados!", icon="✅") # Feedback ao usuário com toast
    except Exception as e:
        logging.exception("Erro ao salvar previsão no banco de dados:") # Log detalhado
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")


def gerar_insights(df):
    """
    Gera insights automáticos sobre a velocidade média, dia mais lento, etc.

    Args:
        df (pd.DataFrame): DataFrame com dados históricos de velocidade processados.

    Returns:
        str: String formatada com os insights.
    """
    insights = []
    if df.empty:
        return "Não há dados para gerar insights neste período."

    media_geral = df['velocidade'].mean()
    insights.append(f"📌 Velocidade média geral: **{media_geral:.2f} km/h**")

    # Encontrar o dia (data específica) com a menor velocidade média dentro do período selecionado
    if 'data' in df.columns and not df['data'].empty:
        # Agrupar por data (apenas a parte da data)
        daily_avg = df.groupby(df['data'].dt.date)['velocidade'].mean()
        if not daily_avg.empty:
            dia_mais_lento_date = daily_avg.idxmin()
            velocidade_dia_mais_lento = daily_avg.min()
            insights.append(f"📅 Dia com a menor velocidade média: **{dia_mais_lento_date.strftime('%d/%m/%Y')}** ({velocidade_dia_mais_lento:.2f} km/h)")
        else:
             insights.append("Não foi possível calcular a velocidade média diária.")
    else:
         insights.append("Coluna 'data' não encontrada ou vazia no DataFrame para insights diários.")


    # Encontrar o dia da semana mais lento em média
    if 'day_of_week' in df.columns and not df['day_of_week'].empty:
            weekday_avg = df.groupby('day_of_week')['velocidade'].mean()
            if not weekday_avg.empty:
                # Mapeamento para português e ordenação
                dias_pt_map = {
                    'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 'Wednesday': 'Quarta-feira',
                    'Thursday': 'Quinta-feira', 'Friday': 'Sexta-feira', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                }
                weekday_avg_pt = weekday_avg.rename(index=dias_pt_map) # <-- Esta é a linha 662 corrigida
                dias_ordenados_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado', 'Domingo']
                weekday_avg_pt = weekday_avg_pt.reindex(dias_ordenados_pt)

                dia_da_semana_mais_lento = weekday_avg_pt.idxmin()
                insights.append(f"🗓️ Dia da semana mais lento (em média): **{dia_da_semana_mais_lento}**")
            else:
                insights.append("Não foi possível calcular a velocidade média por dia da semana.")
    else:
        insights.append("Coluna 'day_of_week' não encontrada ou vazia no DataFrame para insights por dia da semana.")

    # Encontrar a hora do dia mais lenta em média
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

def get_route_metadata():
    """
    Busca metadados completos das rotas incluindo dados históricos e atuais.
    """
    mydb = None
    mycursor = None
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()
        query = """
            SELECT 
                id, name, jam_level, 
                avg_speed, avg_time,
                historic_speed, historic_time 
            FROM routes
            WHERE is_active = 1
        """
        mycursor.execute(query)
        results = mycursor.fetchall()
        col_names = [desc[0] for desc in mycursor.description]
        df = pd.DataFrame(results, columns=col_names)
        return df
    except Exception as e:
        logging.exception("Erro ao obter metadados das rotas:")
        return pd.DataFrame()

def analyze_current_vs_historical(metadata_df):
    """
    Analisa os dados atuais vs históricos e gera insights.
    Retorna DataFrame com métricas calculadas.
    """
    analysis_df = metadata_df.copy()
    
    # Calcular variações percentuais
    analysis_df['var_time'] = ((analysis_df['avg_time'] - analysis_df['historic_time']) / analysis_df['historic_time']) * 100
    analysis_df['var_speed'] = ((analysis_df['avg_speed'] - analysis_df['historic_speed']) / analysis_df['historic_speed']) * 100
    
    # Classificar status
    analysis_df['status'] = np.where(
        (analysis_df['var_time'] > 15) | (analysis_df['var_speed'] < -15),
        'Crítico',
        np.where(
            (analysis_df['var_time'] > 5) | (analysis_df['var_speed'] < -5),
            'Atenção',
            'Normal'
        )
    )


# --- Função Principal do Aplicativo Streamlit ---

def main():
    """
    Função principal que configura a interface do Streamlit, carrega dados
    e exibe análises e previsões.
    """
    # Verificar se as secrets do banco de dados estão configuradas
    if "mysql" not in st.secrets or not all(k in st.secrets["mysql"] for k in ("host", "user", "password", "database")):
        st.error("As credenciais do banco de dados não foram configuradas corretamente no secrets.toml.")
        st.markdown("Por favor, crie ou atualize o arquivo `.streamlit/secrets.toml` na raiz do seu projeto com as informações de conexão do MySQL.")
        logging.error("Secrets do banco de dados não configuradas.") # Log detalhado
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
             logging.warning("Nenhum nome de rota encontrado no banco de dados.") # Log detalhado
             st.stop() # Parar se não houver rotas

        # Usar índice para garantir que o selectbox não quebre se o nome da rota mudar ou não existir
        # Usar session_state para persistir a seleção de rota
        if "main_route_select" not in st.session_state or st.session_state.main_route_select not in all_route_names:
             st.session_state.main_route_select = all_route_names[0]

        try:
            default_main_route_index = all_route_names.index(st.session_state.main_route_select)
        except ValueError:
             default_main_route_index = 0

        route_name = st.selectbox(
            "Rota Principal:",
            all_route_names,
            index=default_main_route_index,
            key="main_route_select_box" # Use um key diferente do session_state key
        )
        # Atualiza o session_state key após o selectbox
        st.session_state.main_route_select = route_name


        compare_enabled = st.checkbox("Comparar com outra rota", key="compare_checkbox")
        second_route = None
        if compare_enabled:
            available_for_comparison = [r for r in all_route_names if r != route_name]
            if available_for_comparison:
                 # Usar session_state para persistir a seleção da rota secundária
                 if "secondary_route_select" not in st.session_state or st.session_state.secondary_route_select not in available_for_comparison:
                      st.session_state.secondary_route_select = available_for_comparison[0]

                 try:
                     default_secondary_route_index = available_for_comparison.index(st.session_state.secondary_route_select)
                 except ValueError:
                      default_secondary_route_index = 0

                 second_route = st.selectbox(
                     "Rota Secundária:",
                     available_for_comparison,
                     index=default_secondary_route_index,
                     key="secondary_route_select_box" # Use um key diferente do session_state key
                 )
                 # Atualiza o session_state key
                 st.session_state.secondary_route_select = second_route

            else:
                 st.info("Não há outras rotas disponíveis para comparação.")
                 compare_enabled = False # Desabilita comparação se não houver outras rotas


        st.subheader("Período de Análise")
        # Usar um seletor de data por rota para flexibilidade na comparação de períodos diferentes
        # Usar session_state para persistir as dates
        today = datetime.date.today()
        week_ago = today - datetime.timedelta(days=7)

        col_date1, col_date2 = st.columns(2)
        with col_date1:
             # Initialize session state for date range if not exists
             if f"date_range_{route_name}" not in st.session_state:
                 st.session_state[f"date_range_{route_name}"] = (week_ago, today)

             date_range_main_input = st.date_input(
                 f"Período para '{route_name}'",
                 value=st.session_state[f"date_range_{route_name}"],
                 max_value=today,
                 key=f"date_range_{route_name}_input" # Use um key diferente
             )
             # Update session state
             st.session_state[f"date_range_{route_name}"] = date_range_main_input
             date_range_main = st.session_state[f"date_range_{route_name}"] # Use o valor persistido


        date_range_secondary = None
        if compare_enabled and second_route:
             with col_date2:
                 # Initialize session state for date range if not exists
                 if f"date_range_{second_route}" not in st.session_state:
                      st.session_state[f"date_range_{second_route}"] = (week_ago, today)

                 date_range_secondary_input = st.date_input(
                      f"Período para '{second_route}'",
                      value=st.session_state[f"date_range_{second_route}"],
                      max_value=today,
                      key=f"date_range_{second_route}_input" # Use um key diferente
                 )
                 # Update session state
                 st.session_state[f"date_range_{second_route}"] = date_range_secondary_input
                 date_range_secondary = st.session_state[f"date_range_{second_route}"] # Use o valor persistido


        # Validar as dates
        if date_range_main and date_range_main[0] > date_range_main[1]:
            st.error("Data final da rota principal não pode ser anterior à data inicial")
            st.stop()
        if compare_enabled and date_range_secondary and date_range_secondary[0] > date_range_secondary[1]:
             st.error("Data final da rota secundária não pode ser anterior à data inicial.")
             st.stop()

        st.subheader("Configurações ARIMA")
        # Adicionar seletor para o período sazonal (m)
        m_period_options = {
            "Diário (480 pontos @ 3min)": 480,
            "Semanal (3360 pontos @ 3min)": 3360,
            "Mensal (~14400 pontos @ 3min)": 14400 # Aproximado
        }
        selected_m_key = st.selectbox(
            "Período sazonal (m) para ARIMA:",
            list(m_period_options.keys()),
            index=0, # Padrão diário
            key="arima_m_select"
        )
        arima_m_period = m_period_options[selected_m_key]

        # Adicionar controle para o número de passos da previsão
        forecast_steps = st.slider(f"Quantos pontos futuros prever ({arima_m_period} / freq 3min)?",
                                   min_value=1, max_value=4 * arima_m_period, # Permite prever até 4 ciclos
                                   value=arima_m_period // 2, # Padrão: meio ciclo
                                   step=int(arima_m_period / 10), # Passo razoável (1/10 do ciclo)
                                   key="forecast_steps_slider")
        st.info(f"Previsão cobrirá aproximadamente {forecast_steps * 3} minutos.")


    st.title("🚀 Análise de Rotas Inteligente")
    st.markdown("Selecione as rotas e o período de análise no painel lateral.")

    routes_info = {}
    routes_to_process = [route_name]
    if compare_enabled and second_route:
        routes_to_process.append(second_route)

    # --- Carregamento e Processamento de Dados ---
    st.header("⏳ Processando Dados...")
    processed_dfs = {} # Dicionário para armazenar os DataFrames processados

    for route in routes_to_process:
        date_range = date_range_main if route == route_name else date_range_secondary
        if date_range is None: # Caso a comparação esteja habilitada, mas a rota secundária não tenha range
             continue

        # Converter objetos date para stringsYYYY-MM-DD para passar para get_data
        start_date_str = date_range[0].strftime('%Y-%m-%d')
        end_date_str = date_range[1].strftime('%Y-%m-%d')

        with st.spinner(f'Carregando e processando dados para {route} de {start_date_str} a {end_date_str}...'):
            # Carregar dados filtrando por nome da rota e período (cached)
            raw_df, error = get_data(
                start_date=start_date_str,
                end_date=end_date_str,
                route_name=route
            )

            if error:
                 st.error(f"Erro ao carregar dados para {route}: {error}")
                 logging.error(f"Erro ao carregar dados para {route}: {error}")
                 routes_info[route] = {'data': pd.DataFrame(), 'id': None, 'error': error}
                 continue # Pula para a próxima rota se houver erro

            if raw_df.empty:
                st.warning(f"Nenhum dado encontrado para a rota '{route}' no período de {start_date_str} a {end_date_str}. Por favor, ajuste o intervalo de dates.")
                logging.warning(f"Nenhum dado encontrado para a rota '{route}' no período {start_date_str} a {end_date_str}.")
                routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                continue # Pula para a próxima rota

            # Adicionar indicador de qualidade dos dados (dados ausentes)
            total_records = len(raw_df)
            initial_nulls = raw_df['velocidade'].isnull().sum()
            initial_null_percentage = (initial_nulls / total_records) * 100 if total_records > 0 else 0
            st.metric(f"Dados Ausentes Inicialmente ({route})", f"{initial_null_percentage:.1f}%")

            # Obter o ID da rota (assumindo que há apenas um ID por nome no período selecionado)
            try:
                 route_id = raw_df['route_id'].iloc[0]
            except IndexError:
                 st.error(f"Não foi possível obter o ID da rota para '{route}'. Dados insuficientes.")
                 logging.error(f"Não foi possível obter ID da rota para '{route}'. DataFrame vazio ou sem route_id.")
                 routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                 continue

            # Limpar e processar os dados
            processed_df = clean_data(raw_df)

            if processed_df.empty:
                 # Mensagem de warning já exibida dentro de clean_data se todos os valores forem nulos
                 routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                 continue


            routes_info[route] = {
                'data': processed_df,
                'id': route_id
            }
            processed_dfs[route] = processed_df # Armazena para comparação
        st.toast(f"Dados para {route} carregados e processados ({len(processed_df)} registros).", icon="✅") # Feedback com toast


    # --- Seção de Visualização ---

    # Se não houver dados carregados para nenhuma rota, parar por aqui
    if not routes_info or all(info['data'].empty for info in routes_info.values()):
         st.info("Selecione as rotas e um período com dados disponíveis no painel lateral para continuar.")
         return # Sai da função main se não houver dados


    st.header("🗺️ Visualização Geográfica")
    # O mapa é exibido por rota dentro do loop de processamento
    for route in routes_to_process:
         # Verifica se a rota foi carregada com sucesso e tem dados
         if route in routes_info and not routes_info[route]['data'].empty:
             route_id = routes_info[route]['id']
             # O expander deve ser dentro do loop para que cada rota tenha seu mapa
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

                      # Determinar um zoom inicial razoável baseado nos bounds (heurística simples)
                      # Calcula a extensão longitudinal e ajusta o zoom
                      lon_extent = max_lon - min_lon
                      lat_extent = max_lat - min_lat
                      # Fórmula de zoom aproximada (ajuste conforme necessário)
                      if lon_extent > 0 and lat_extent > 0:
                         zoom_lon = 360 / lon_extent
                         zoom_lat = 180 / lat_extent
                         zoom = min(zoom_lon, zoom_lat) * 0.5 # Ajuste o fator (0.5)
                         zoom = min(max(zoom, 10), 15) # Limita o zoom entre 10 e 15
                      else:
                         zoom = 12 # Zoom padrão se a rota for muito pequena ou um ponto

                      fig = go.Figure(go.Scattermapbox(
                          mode="lines+markers",
                          lon=route_coords['longitude'],
                          lat=route_coords['latitude'],
                          marker={'size': 8, 'color': ACCENT_COLOR},
                          line=dict(width=4, color=PRIMARY_COLOR),
                          hovertext=[f"Ponto {i+1}" for i in range(len(route_coords))],
                          hoverinfo="text+lat+lon" # Mostra texto customizado + lat/lon no tooltip
                      ))

                      fig.update_layout(
                          mapbox={
                              'style': "carto-darkmatter", # Estilo de mapa que combina com o tema escuro
                              'center': {'lat': center_lat, 'lon': center_lon},
                              'zoom': zoom,
                              # bounds podem ser usados para focar na área
                              'bounds': {'west': min_lon - lon_buffer, 'east': max_lon + lon_buffer,
                                         'south': min_lat - lat_buffer, 'north': max_lat + lat_buffer}
                          },
                          margin={"r":0,"t":0,"l":0,"b":0},
                          height=500, # Altura do mapa
                          plot_bgcolor=SECONDARY_BACKGROUND_COLOR, # Fundo do plot
                          paper_bgcolor=SECONDARY_BACKGROUND_COLOR, # Fundo do papel (figura)
                          font=dict(color=TEXT_COLOR), # Cor da fonte global do gráfico
                          title=f"Mapa da Rota: {route}" # Adiciona título ao mapa
                      )
                      st.plotly_chart(fig, use_container_width=True)
                  else:
                      st.warning(f"Nenhuma coordenada geográfica encontrada para a rota '{route}'. Não é possível exibir o mapa.")
         elif route in routes_info and 'error' in routes_info[route]:
              st.warning(f"Mapa não disponível para '{route}' devido a erro no carregamento de dados.")
         else:
             # Isso pode acontecer se compare_enabled for True mas a segunda rota não puder ser carregada
             st.info(f"Dados insuficientes para exibir o mapa da rota '{route}'.")

    st.header("📈 Análise de Momento: Histórico vs Atual")
    
    # Carregar metadados das rotas
    route_metadata = get_route_metadata()
    if not route_metadata.empty:
        analysis_df = analyze_current_vs_historical(route_metadata)
        
        with st.expander("🔍 Principais Observações", expanded=True):
            st.markdown("""
            **Relação Tempo vs Velocidade:**
            - Quando avg_time > historic_time: Redução de velocidade (congestionamento)
            - Quando avg_time < historic_time: Melhoria no fluxo
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rotas Críticas", 
                         len(analysis_df[analysis_df['status'] == 'Crítico']))
            with col2:
                avg_delay = analysis_df['var_time'].mean()
                st.metric("Atraso Médio", f"{avg_delay:.1f}%")
        
        with st.expander("🚦 Top 5 Rotas com Maiores Discrepâncias"):
            top_criticas = analysis_df[analysis_df['status'] == 'Crítico'].head(5)
            if not top_criticas.empty:
                for idx, row in top_criticas.iterrows():
                    st.markdown(f"""
                    **{row['name']}**
                    - 🔴 Tempo Atual: {row['avg_time']}s (Histórico: {row['historic_time']}s)
                    - 🚗 Velocidade Atual: {row['avg_speed']}km/h (Histórico: {row['historic_speed']}km/h)
                    - 📈 Variação: +{row['var_time']:.1f}% tempo | {row['var_speed']:.1f}% velocidade
                    """)
            else:
                st.info("Nenhuma rota crítica identificada")
        
        with st.expander("📊 Análise Detalhada por Categoria"):
            st.dataframe(
                analysis_df[['name', 'status', 'avg_time', 'historic_time', 
                           'avg_speed', 'historic_speed', 'var_time', 'var_speed']],
                column_config={
                    "name": "Rota",
                    "status": st.column_config.SelectboxColumn(
                        "Status",
                        options=["Normal", "Atenção", "Crítico"]
                    ),
                    "avg_time": "Tempo Atual (s)",
                    "historic_time": "Tempo Histórico (s)",
                    "avg_speed": "Velocidade Atual (km/h)",
                    "historic_speed": "Velocidade Histórica (km/h)",
                    "var_time": st.column_config.ProgressColumn(
                        "Variação Tempo",
                        format="+%.1f%%",
                        min_value=-100,
                        max_value=300
                    ),
                    "var_speed": st.column_config.ProgressColumn(
                        "Variação Velocidade",
                        format="%.1f%%",
                        min_value=-100,
                        max_value=100
                    )
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning("Não foi possível carregar metadados das rotas")


    st.header("📊 Visualização de Dados Históricos")

    # --- Comparação Visual de Dados Históricos (Gráfico de Linha Plotly) ---
    if len(processed_dfs) > 0:
         st.subheader("Comparação de Velocidade Histórica ao Longo do Tempo")
         fig_historical_comparison = go.Figure()

         colors = [PRIMARY_COLOR, ACCENT_COLOR] # Cores para as rotas

         for i, (r_name, r_df) in enumerate(processed_dfs.items()):
              if not r_df.empty:
                   fig_historical_comparison.add_trace(go.Scatter(
                       x=r_df['data'],
                       y=r_df['velocidade'],
                       mode='lines',
                       name=f'Histórico: {r_name}',
                       line=dict(color=colors[i % len(colors)], width=2) # Usa cores distintas
                   ))
              else:
                   st.info(f"Dados insuficientes para incluir '{r_name}' no gráfico de comparação histórica.")


         if len(fig_historical_comparison.data) > 0: # Exibe apenas se houver pelo menos uma rota
              fig_historical_comparison.update_layout(
                  title='Velocidade Histórica ao Longo do Tempo',
                  xaxis_title="Data/Hora",
                  yaxis_title="Velocidade (km/h)",
                  hovermode='x unified',
                  plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                  paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                  font=dict(color=TEXT_COLOR),
                  title_font_color=TEXT_COLOR,
                  xaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font_color=TEXT_COLOR),
                  yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font_color=TEXT_COLOR),
                  legend=dict(font=dict(color=TEXT_COLOR))
              )
              st.plotly_chart(fig_historical_comparison, use_container_width=True)
         elif compare_enabled:
              st.info("Dados insuficientes para realizar a comparação histórica entre as rotas selecionadas.")


    # --- Seção de Análise Preditiva ---
    st.header("📈 Análise Preditiva")
    for route in routes_to_process:
        # Verifica se a rota foi carregada com sucesso e tem dados processados
        if route in routes_info and not routes_info[route]['data'].empty:
            processed_df = routes_info[route]['data']
            route_id = routes_info[route]['id']

            # Expander para cada rota
            with st.expander(f"Análise para {route}", expanded=True):

                st.subheader("🧠 Insights Automáticos")
                st.markdown(gerar_insights(processed_df))

                st.subheader("📉 Decomposição Temporal")
                # Passa o df processado que clean_data retornou
                # Esta função usa Matplotlib, a cor do tema é configurada DENTRO dela.
                seasonal_decomposition_plot(processed_df)


                st.subheader("🔥 Heatmap Horário por Dia da Semana")
                if not processed_df.empty:
                    pivot_table = processed_df.pivot_table(
                        index='day_of_week',
                        columns='hour',
                        values='velocidade',
                        aggfunc='mean'
                    )

                    # Reordenar dias da semana (em português)
                    dias_ordenados_eng = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    dias_pt = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                    dia_mapping = dict(zip(dias_ordenados_eng, dias_pt))

                    # Reindexar a tabela pivotada para garantir a ordem dos dias
                    pivot_table = pivot_table.reindex(dias_ordenados_eng)
                    # Renomear o índice para português
                    pivot_table.index = pivot_table.index.map(dia_mapping)


                    # --- Usar Matplotlib/Seaborn Heatmap ---
                    # Criar uma figura e eixos Matplotlib
                    fig_mpl, ax_mpl = plt.subplots(figsize=(12, 8)) # Tamanho da figura

                    # Gerar o heatmap usando Seaborn
                    sns.heatmap(
                        pivot_table,
                        annot=True,      # Mostrar os valores nas células
                        fmt=".0f",       # Formatar os valores para 0 casas decimais (inteiro) <--- Corrigido para 0 casas decimais
                        cmap="viridis",  # Mapa de cores (similar ao Viridis do Plotly)
                        linewidths=.5,   # Adicionar linhas entre as células para clareza
                        ax=ax_mpl        # Desenhar no eixo Matplotlib criado
                         # annot_kws={"color": TEXT_COLOR} # Opcional: cor da fonte da anotação (pode prejudicar leitura)
                    )

                    # Configurar títulos e labels dos eixos para o tema escuro
                    ax_mpl.set_title('Velocidade Média por Dia da Semana e Hora', color=TEXT_COLOR)
                    ax_mpl.set_xlabel('Hora do Dia', color=TEXT_COLOR)
                    ax_mpl.set_ylabel('Dia da Semana', color=TEXT_COLOR)

                    # Configurar cor dos ticks dos eixos e fundo do plot
                    ax_mpl.tick_params(axis='x', colors=TEXT_COLOR)
                    ax_mpl.tick_params(axis='y', colors=TEXT_COLOR)
                    ax_mpl.set_facecolor(SECONDARY_BACKGROUND_COLOR) # Fundo da área do plot

                    # Configurar cor de fundo da figura inteira
                    fig_mpl.patch.set_facecolor(SECONDARY_BACKGROUND_COLOR) # Fundo da figura

                    # Configurar a cor da barra de cor (colorbar)
                    cbar = ax_mpl.collections[0].colorbar # Obter o objeto colorbar
                    if cbar:
                         cbar.ax.tick_params(colors=TEXT_COLOR) # Cor dos ticks
                         cbar.set_label('Velocidade Média (km/h)', color=TEXT_COLOR) # Cor do label

                    # Exibir a figura Matplotlib no Streamlit
                    st.pyplot(fig_mpl)
                    plt.close(fig_mpl) # Fechar a figura para liberar memória

                else:
                    st.info("Dados insuficientes para gerar o Heatmap.")


                st.subheader("🔮 Previsão de Velocidade (ARIMA)")

                # Botão para rodar a previsão (usa forecast_steps e arima_m_period definidos na sidebar)
                if st.button(f"Gerar Previsão para {route}", key=f"generate_forecast_{route}"):
                     forecast_df = pd.DataFrame() # Initialize DataFrame

                     # --- Try/Except para a Geração da Previsão ARIMA ---
                     try:
                         st.info(f"Iniciando geração da previsão ARIMA para {route} com período sazonal m={arima_m_period} e {forecast_steps} passos futuros...")
                         # Chamada da função de previsão ARIMA (agora com exógenas e m_period)
                         forecast_df = create_arima_forecast(processed_df, route_id, steps=forecast_steps, m_period=arima_m_period)

                         if not forecast_df.empty:
                             st.success(f"Previsão gerada para os próximos {forecast_steps * 3} minutos.")
                             st.toast("Previsão gerada!", icon="✅") # Feedback com toast


                             # --- Try/Except para Plotar o Gráfico de Previsão ---
                             try:
                                 st.info("Gerando gráfico de previsão...")
                                 fig_forecast = go.Figure()

                                 # Adiciona os dados históricos
                                 fig_forecast.add_trace(go.Scatter(
                                     x=processed_df['data'],
                                     y=processed_df['velocidade'],
                                     mode='lines',
                                     name=f'Histórico: {route}', # Nome da rota no histórico
                                     line=dict(color=TEXT_COLOR, width=2) # Cor para o histórico
                                 ))

                                 # Adiciona a previsão
                                 fig_forecast.add_trace(go.Scatter(
                                     x=forecast_df['ds'],
                                     y=forecast_df['yhat'],
                                     mode='lines',
                                     name=f'Previsão: {route}', # Nome da rota na previsão
                                     line=dict(color=PRIMARY_COLOR, width=3) # Cor primária para a previsão
                                 ))

                                 # Adiciona o intervalo de confiança
                                 fig_forecast.add_trace(go.Scatter(
                                     x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]), # Dates para o polígono (ida e volta)
                                     y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]), # Limites (superior e inferior invertido)
                                     fill='toself', # Preenche a área entre as duas linhas
                                     fillcolor='rgba(0, 175, 255, 0.2)', # Cor semi-transparente (similar ao PRIMARY_COLOR)
                                     line=dict(color='rgba(255,255,255,0)'), # Linha invisível
                                     name=f'Intervalo de Confiança 95% ({route})'
                                 ))

                                 # Configura o layout do gráfico de previsão
                                 fig_forecast.update_layout(
                                     title=f'Previsão de Velocidade para {route}',
                                     xaxis_title="Data/Hora",
                                     yaxis_title="Velocidade (km/h)",
                                     hovermode='x unified', # Agrupa tooltips por eixo X
                                     plot_bgcolor=SECONDARY_BACKGROUND_COLOR,
                                     paper_bgcolor=SECONDARY_BACKGROUND_COLOR,
                                     font=dict(color=TEXT_COLOR),
                                     title_font_color=TEXT_COLOR,
                                     xaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font_color=TEXT_COLOR),
                                     yaxis=dict(tickfont=dict(color=TEXT_COLOR), title_font_color=TEXT_COLOR),
                                     legend=dict(font=dict(color=TEXT_COLOR))
                                 )

                                 st.plotly_chart(fig_forecast, use_container_width=True)
                                 st.success("Gráfico de previsão gerado.")


                             except Exception as e:
                                 logging.exception("Erro ao gerar ou exibir o gráfico de previsão:") # Log detalhado
                                 st.error(f"Erro ao gerar ou exibir o gráfico de previsão: {e}")
                                 st.info("Verifique se há dados suficientes na previsão gerada ou se há problemas na configuração do gráfico Plotly.")

                             # --- Try/Except para Salvar no Banco de Dados ---
                             # O botão de salvar só aparece APÓS a previsão ser gerada e plotada
                             if st.button(f"Salvar Previsão no Banco de Dados para {route}", key=f"save_forecast_{route}"):
                                  save_forecast_to_db(forecast_df)


                         else:
                             st.warning("Previsão não gerada ou DataFrame de previsão vazio. Não é possível exibir o gráfico ou salvar.")
                             st.toast("Previsão falhou!", icon="❌") # Feedback com toast

                     except Exception as e:
                         logging.exception("Erro fatal durante a geração da previsão ARIMA:") # Log detalhado
                         st.error(f"Erro fatal durante a geração da previsão ARIMA: {e}")
                         st.info("Verifique os dados de entrada, a quantidade de dados, ou a configuração do modelo ARIMA.")
                         st.toast("Previsão falhou!", icon="❌") # Feedback com toast


                # Mensagem inicial antes de gerar a previsão
                elif f"generate_forecast_{route}" not in st.session_state:
                    st.info("Configure o período sazonal e os passos futuros na sidebar e clique em 'Gerar Previsão'.")


        # Adiciona uma linha separadora entre as análises de rotas se houver mais de uma
        if len(routes_to_process) > 1 and routes_to_process.index(route) < len(routes_to_process) - 1:
            st.markdown("---") # Linha horizontal

    # Mensagem final caso nenhuma rota tenha dados
    if not routes_info or all(info['data'].empty for info in routes_info.values()):
         st.info("Nenhuma análise exibida. Selecione rotas com dados disponíveis.")


# --- Executa o aplicativo Streamlit ---
if __name__ == "__main__":
    main()