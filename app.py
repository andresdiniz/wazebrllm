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
import plotly as px
import plotly.graph_objects as go
from io import BytesIO
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error
import datetime # Importar datetime para manipular datas
import plotly

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
/* Melhorar aparência do selectbox */
.stSelectbox > div[data-baseweb="select"] > div {{
     background-color: var(--secondary-background-color);
     color: var(--text-color);
     border: 1px solid #555;
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

# faz conexxão com o banco de dados MySQL (cached)
@st.cache_resource # Usar cache_resource para conexões de DB
def get_db_connection():
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
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.stop() # Parar a execução se não conseguir conectar


# Carregar apenas nomes das rotas (cached)
@st.cache_data # Usar cache_data para dados estáticos como nomes de rotas
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
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

@st.cache_data # Usar cache_data para os dados históricos, dependendo dos parâmetros
def get_data(start_date=None, end_date=None, route_name=None):
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
        return pd.DataFrame(), str(e) # Retorna DataFrame vazio e erro
    finally:
        if mycursor:
            mycursor.close()
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

@st.cache_data # Usar cache_data para coordenadas de rota
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
        # Não feche a conexão 'mydb' aqui, pois ela é gerenciada por st.cache_resource

# --- Funções de Processamento e Análise ---

# Esta função processa o DataFrame e pode ser chamada após carregar os dados
def clean_data(df):
    df = df.copy()
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
    # locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8') # Configurar localidade (pode precisar instalar no ambiente)
    df['day_of_week'] = df['data'].dt.day_name() # Retorna em inglês por padrão, mapearemos para o heatmap
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade']) # Remove linhas onde a velocidade ainda é NaN


# Função de decomposição sazonal (revisada para usar índice de tempo e frequência)
def seasonal_decomposition_plot(df):
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
    period = 480

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
         st.warning(f"Não foi possível realizar a decomposição sazonal: {e}")
         st.info("Verifique se os dados têm uma frequência regular ou se há dados suficientes.")


# Função de previsão ARIMA (revisada para usar intervalos de confiança e tratamento de dados)
# Não cacheamos previsões pois elas dependem de dados recentes e podem ser acionadas pelo usuário
# @st.cache_data # Não use cache_data para previsões se elas devem ser geradas sob demanda
def create_arima_forecast(df, route_id, steps=10):
    if df.empty:
        return pd.DataFrame()

    # Preparar dados para auto_arima (já vem limpo)
    # Garantir frequência temporal, interpolando se houver lacunas curtas
    arima_data = df.set_index('data')['velocidade'].asfreq('3min').dropna()

    # Precisa de dados suficientes para o modelo sazonal ARIMA
    # Uma semana de dados (3min freq) = 480 pontos/dia * 7 dias = 3360 pontos
    min_data_points = 24 * 7 * (60/3) # Mínimo ~1 semana de dados com freq de 3min
    if len(arima_data) < min_data_points:
         st.warning(f"Dados insuficientes ({len(arima_data)} pontos) para um modelo de previsão ARIMA sazonal robusto. Necessário mais dados históricos (ex: pelo menos 1 semana com frequência de 3min, aproximadamente {int(min_data_points)} pontos).")
         return pd.DataFrame()

    try:
        # auto_arima encontrará os melhores parâmetros p,d,q,P,D,Q
        # m=480 para sazonalidade diária em dados de 3 em 3 minutos
        # Adicionado stepwise=True para acelerar, n_fits para limitar tentativas, random_state para reprodutibilidade
        with st.spinner(f"Treinando modelo ARIMA para a rota {route_id}..."):
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
        # Usando credenciais do secrets
        engine = create_engine(
            f'mysql+mysqlconnector://{st.secrets["mysql"]["user"]}:{st.secrets["mysql"]["password"]}@{st.secrets["mysql"]["host"]}/{st.secrets["mysql"]["database"]}'
        )
        # Usando o gerenciador de contexto do SQLAlchemy para garantir commit/rollback e fechar a conexão
        # if_exists='append' adiciona novas linhas. Se você precisar evitar duplicatas,
        # pode precisar de uma lógica de upsert ou verificar antes de inserir.
        with engine.begin() as connection:
             forecast_df_mapped.to_sql('forecast_history', con=connection, if_exists='append', index=False)
             st.success("Previsão salva no banco de dados!") # Feedback ao usuário
    except Exception as e:
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")

# Função de geração de insights automáticos
def gerar_insights(df):
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
            weekday_avg_pt = weekday_avg.rename(index=dias_pt_map)
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

# --- Função Principal do Aplicativo Streamlit ---

def main():
    # Verificar se as secrets do banco de dados estão configuradas
    if "mysql" not in st.secrets or not all(k in st.secrets["mysql"] for k in ("host", "user", "password", "database")):
        st.error("As credenciais do banco de dados não foram configuradas corretamente no secrets.toml.")
        st.markdown("Por favor, crie ou atualize o arquivo `.streamlit/secrets.toml` na raiz do seu projeto com as informações de conexão do MySQL.")
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
        # Usar session_state para persistir as datas
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


        # Validar as datas
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
                 routes_info[route] = {'data': pd.DataFrame(), 'id': None, 'error': error}
                 continue # Pula para a próxima rota se houver erro

            if raw_df.empty:
                st.warning(f"Nenhum dado encontrado para a rota '{route}' no período de {start_date_str} a {end_date_str}. Por favor, ajuste o intervalo de datas.")
                routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                continue # Pula para a próxima rota

            # Obter o ID da rota (assumindo que há apenas um ID por nome no período selecionado)
            # Se houver dados, deve haver um route_id
            try:
                 route_id = raw_df['route_id'].iloc[0]
            except IndexError:
                 st.error(f"Não foi possível obter o ID da rota para '{route}'.")
                 routes_info[route] = {'data': pd.DataFrame(), 'id': None}
                 continue

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


    # --- Seção de Análise ---
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
                    # O mapeamento do índice para português será feito DEPOIS de resetar o índice


                    # --- CORREÇÃO DO KEYERROR APLICADA AQUI ---
                    # Resetar o índice para transformar a coluna 'day_of_week' (o índice original) em uma coluna
                    pivot_table_reset = pivot_table.reset_index()
                    # Renomear a coluna que era o índice ('day_of_week') para 'Dia da Semana'
                    # ESTA LINHA FOI CORRIGIDA DE {'index': 'Dia da Semana'} PARA {'day_of_week': 'Dia da Semana'}
                    pivot_table_reset = pivot_table_reset.rename(columns={'day_of_week': 'Dia da Semana'})

                    # Aplicar o mapeamento dos nomes dos dias para português AGORA que 'Dia da Semana' é uma coluna
                    pivot_table_reset['Dia da Semana'] = pivot_table_reset['Dia da Semana'].map(dia_mapping)

                    # Define uma categoria para a coluna 'Dia da Semana' para garantir a ordem correta no gráfico
                    # Isso também ajuda Plotly a entender a ordem do eixo Y
                    pivot_table_reset['Dia da Semana'] = pd.Categorical(
                        pivot_table_reset['Dia da Semana'], categories=dias_pt, ordered=True
                    )
                    # Opcional: Reordenar o dataframe pelo dia da semana categórico (útil para depuração, mas o Plotly geralmente respeita a ordem categórica)
                    # Removido a reordenação explícita aqui para simplificar, a categoria já deve bastar.
                    # pivot_table_reset = pivot_table_reset.sort_values('Dia da Semana')


                    # Obter a lista de colunas de hora (todas as colunas exceto 'Dia da Semana')
                    # Estes são os nomes das colunas numéricas 0, 1, 2, ...
                    hour_columns = [col for col in pivot_table_reset.columns if col != 'Dia da Semana']

                    # --- DIAGNÓSTICO FINAL ANTES DO MELT (Opcional) ---
                    # st.write("DEBUG (Before Melt): pivot_table_reset columns:", pivot_table_reset.columns.tolist())
                    # st.write("DEBUG (Before Melt): pivot_table_reset dtypes:", pivot_table_reset.dtypes)
                    # st.write("DEBUG (Before Melt): pivot_table_reset head():", pivot_table_reset.head())
                    # st.write("DEBUG (Before Melt): id_vars for melt:", ['Dia da Semana'])
                    # st.write("DEBUG (Before Melt): value_vars (hour_columns) for melt:", hour_columns)
                    # --- FIM DIAGNÓSTICO ---


                    # Derreter o DataFrame, especificando explicitamente as colunas de valor (as horas)
                    # ESTA LINHA FOI CORRIGIDA PARA USAR value_vars=hour_columns
                    melted_heatmap_data = pivot_table_reset.melt(
                        id_vars=['Dia da Semana'],        # Coluna(s) para manter como identificadores
                        value_vars=hour_columns,         # ESPECIFICAR AS COLUNAS DE HORA
                        var_name='Hora do Dia',          # Nome para a nova coluna de variáveis
                        value_name='Velocidade Média'    # Nome para a nova coluna de valores
                    )

                    # Garantir que a coluna de hora seja numérica (Plotly gosta disso para eixos numéricos)
                    melted_heatmap_data['Hora do Dia'] = pd.to_numeric(melted_heatmap_data['Hora do Dia'])

                    # --- DIAGNÓSTICO FINAL ANTES DO HEATMAP ---
                    st.write("DEBUG (Heatmap Data): melted_heatmap_data columns:", melted_heatmap_data.columns.tolist())
                    st.write("DEBUG (Heatmap Data): melted_heatmap_data dtypes:", melted_heatmap_data.dtypes)
                    st.write("DEBUG (Heatmap Data): melted_heatmap_data head():", melted_heatmap_data.head())
                    # st.write("DEBUG (Heatmap Data): melted_heatmap_data describe():", melted_heatmap_data.describe()) # Descomente se precisar de estatísticas
                    st.write("DEBUG (Heatmap Data): melted_heatmap_data isnull().sum():", melted_heatmap_data.isnull().sum())
                    # --- FIM DIAGNÓSTICO FINAL ---


                    # --- Plotly Heatmap Code usando dados derretidos ---
                    # Agora especificamos explicitamente x, y, e z
                    # ESTA É A LINHA QUE ESTÁ CAUSANDO O ATTRIBUTEERROR (line 889)
                    fig_heatmap = px.heatmap(
                        melted_heatmap_data, # Passa o DataFrame derretido
                        x='Hora do Dia',     # Nome da coluna para o eixo X
                        y='Dia da Semana',   # Nome da coluna para o eixo Y
                        z='Velocidade Média',# Nome da coluna para os valores/cor
                        text_auto=True,      # Mostra o valor dentro da célula (opcional) - Plotly 5.x+
                        aspect="auto",
                        title="Velocidade Média por Dia da Semana e Hora",
                        color_continuous_scale="Viridis" # Use o mesmo cmap ou similar ao viridis
                    )

                    # Configurar layout para combinar com o tema (cores, fontes)
                    fig_heatmap.update_layout(
                        title_font_color=TEXT_COLOR,
                        xaxis=dict(tickfont=dict(color=TEXT_COLOR), title="Hora do Dia", title_font_color=TEXT_COLOR),
                        yaxis=dict(tickfont=dict(color=TEXT_COLOR), title="Dia da Semana", title_font_color=TEXT_COLOR),
                        plot_bgcolor=SECONDARY_BACKGROUND_COLOR, # Fundo do plot
                        paper_bgcolor=SECONDARY_BACKGROUND_COLOR, # Fundo do papel (figura)
                        font=dict(color=TEXT_COLOR) # Cor da fonte global do gráfico
                    )
                    # O tooltip com o valor aparece por padrão ao passar o mouse com px.heatmap quando x, y, z são especificados


                    # Exibe o gráfico Plotly no Streamlit
                    st.plotly_chart(fig_heatmap, use_container_width=True)

                else:
                    st.info("Dados insuficientes para gerar o Heatmap.")


                st.subheader("🔮 Previsão de Velocidade (ARIMA)")
                # Adicionar controle para o número de passos da previsão
                forecast_steps = st.slider(f"Quantos pontos futuros prever (rota: {route})?", min_value=1, max_value=48 * (60//3), value=48 * (60//3)//2, step=(60//3), key=f"forecast_steps_{route}") # Prever até 48 horas em passos de 3min

                # Botão para rodar a previsão
                if st.button(f"Gerar Previsão para {route}", key=f"generate_forecast_{route}"):
                     forecast_df = create_arima_forecast(processed_df, route_id, steps=forecast_steps)

                     if not forecast_df.empty:
                         st.success(f"Previsão gerada para os próximos {forecast_steps * 3} minutos.")

                         # Visualizar a previsão
                         fig_forecast = go.Figure()

                         # Adiciona os dados históricos
                         fig_forecast.add_trace(go.Scatter(
                             x=processed_df['data'],
                             y=processed_df['velocidade'],
                             mode='lines',
                             name='Histórico',
                             line=dict(color=TEXT_COLOR, width=2) # Cor para o histórico
                         ))

                         # Adiciona a previsão
                         fig_forecast.add_trace(go.Scatter(
                             x=forecast_df['ds'],
                             y=forecast_df['yhat'],
                             mode='lines',
                             name='Previsão',
                             line=dict(color=PRIMARY_COLOR, width=3) # Cor primária para a previsão
                         ))

                         # Adiciona o intervalo de confiança
                         fig_forecast.add_trace(go.Scatter(
                             x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]), # Datas para o polígono (ida e volta)
                             y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]), # Limites (superior e inferior invertido)
                             fill='toself', # Preenche a área entre as duas linhas
                             fillcolor='rgba(0, 175, 255, 0.2)', # Cor semi-transparente (similar ao PRIMARY_COLOR)
                             line=dict(color='rgba(255,255,255,0)'), # Linha invisível
                             name='Intervalo de Confiança 95%'
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

                         # Botão para salvar a previsão no banco de dados
                         if st.button(f"Salvar Previsão no Banco de Dados para {route}", key=f"save_forecast_{route}"):
                              save_forecast_to_db(forecast_df)
                     else:
                         st.info("Previsão não gerada. Verifique os dados ou o período selecionado.")

                elif f"generate_forecast_{route}" not in st.session_state:
                    # Mensagem inicial antes de gerar a previsão
                    st.info("Clique no botão acima para gerar a previsão de velocidade para esta rota.")

        # Adiciona uma linha separadora entre as análises de rotas se houver mais de uma
        if len(routes_to_process) > 1 and routes_to_process.index(route) < len(routes_to_process) - 1:
            st.markdown("---") # Linha horizontal

    # Mensagem final caso nenhuma rota tenha dados
    if not routes_info or all(info['data'].empty for info in routes_info.values()):
         st.info("Nenhuma análise exibida. Selecione rotas com dados disponíveis.")


# --- Executa o aplicativo Streamlit ---
if __name__ == "__main__":
    main()