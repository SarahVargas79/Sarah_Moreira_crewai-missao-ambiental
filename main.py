from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process, LLM

import os
import sys
import io

# Força saída UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 

os.environ["GOOGLE_API_KEY"] = "AIzaSyB-SH8V5QQ0QCtNXsP1QRJ87ztA2vp2J04"

# Instanciando o modelo de linguagem da Google (Gemini 2.0)
llm = LLM(
    model='gemini/gemini-2.0-flash-lite',
    verbose=True,          # Mostra logs detalhados no terminal
    # Controla a criatividade das respostas (0 = mais preciso)
    temperature=0.4,
    api_key = os.environ["GOOGLE_API_KEY"]
)

# ---------------------
# CRIAÇÃO DOS AGENTES
# ---------------------

# Agente 1: Pesquisador Ambiental
# Responsável por buscar informações sobre como jogos podem ensinar sustentabilidade
pesquisador = Agent(
    role="Pesquisador Ambiental",
    goal="Investigar formas de promover a conscientização ambiental em jovens através de jogos",
    backstory="Especialista em educação ambiental, apaixonado por tecnologias e jogos educativos",
    verbose=True,
    llm=llm
)

# Agente 2: Designer de Missão Interativa
# Irá criar uma missão gamificada com base nas pesquisas
designer = Agent(
    role="Designer de Missão Interativa",
    goal="Transformar os dados pesquisados em uma missão de jogo estilo The Sims com foco em conscientização",
    backstory="Criador de jogos independentes e narrativas interativas voltadas para mudanças sociais",
    verbose=True,
    llm=llm
)

# Agente 3: Revisor de Conteúdo Gamificado
# Irá revisar e melhorar o roteiro da missão para torná-la mais clara e divertida
revisor = Agent(
    role="Revisor de Conteúdo Gamificado",
    goal="Aprimorar a missão para torná-la clara, educativa e divertida",
    backstory="Entusiasta de games educativos e revisão de conteúdo lúdico para engajamento juvenil",
    verbose=True,
    llm=llm
)

# ---------------------
# DEFINIÇÃO DAS TAREFAS
# ---------------------

# Tarefa 1: Pesquisa sobre uso de jogos para educação ambiental
tarefa_pesquisa = Task(
    description="Pesquisar como jogos podem ser usados para promover a conscientização ambiental entre jovens",
    expected_output="Um resumo com pelo menos 3 estratégias ou abordagens eficazes, em português",
    agent=pesquisador
)

# Tarefa 2: Criação da missão gamificada baseada nas descobertas da pesquisa
tarefa_design = Task(
    description="Com base na pesquisa anterior, crie uma missão interativa estilo The Sims com objetivo de ensinar práticas ambientais",
    expected_output="Roteiro de missão contendo narrativa, objetivos do jogador e mecânicas básicas",
    agent=designer
)

# Tarefa 3: Revisar e melhorar o roteiro da missão
tarefa_revisao = Task(
    description="Revisar a missão criada, melhorar a narrativa, ajustar a linguagem e garantir apelo educativo e divertido",
    expected_output="Versão revisada e coesa da missão gamificada, pronta para prototipagem",
    agent=revisor
)

# ---------------------
# EXECUÇÃO DA CREW
# ---------------------

# Junta todos os agentes e tarefas em um processo sequencial
crew = Crew(
    agents=[pesquisador, designer, revisor],      # Lista de agentes
    tasks=[tarefa_pesquisa, tarefa_design, tarefa_revisao],  # Lista de tarefas
    # Define que as tarefas serão executadas em ordem
    process=Process.sequential,
    verbose=True
)

# Inicia o sistema multiagente
resultado = crew.kickoff()

# Exibe o resultado final da execução no terminal
print(resultado)
