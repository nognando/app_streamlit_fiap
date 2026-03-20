import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da página
st.set_page_config(page_title="Avaliador de Desempenho", page_icon="🎓", layout="wide")

# ==========================================
# 1. Carregamento do Modelos
# ==========================================

modelo = joblib.load('modelo.joblib')
features = joblib.load('modelo_features.joblib')
        

# Mapeamentos Manuais (Se você não tiver o Encoder salvo do treino)
# Isso traduz o que o usuário escolhe para o que o modelo entende.
mapeamento_fases = {"Fase 1": 1, "Fase 2": 2, "Fase 3": 3, "Fase 4": 4, "Fase 5": 5}
mapeamento_pedras = {"Ágata": 0, "Ametista": 1, "Quartzo": 2, "Topázio": 3} # Exemplos
mapeamento_genero = {"F": 0, "M": 1}
mapeamento_instituicao = {"Escola Pública": 0, "Escola Privada": 1}

# Carregar o modelo
#features = carregar_recursos()

# ==========================================
# 2. Interface do Usuário (Formulário)
# ==========================================
st.title("🎓 Formulário de Avaliação do Aluno")
st.markdown("Preencha os dados abaixo para receber a previsibilidade de Risco de Defazagem do Aluno.")

# Se o modelo não carregou, para a execução
if features is None:
    st.stop()

# Criando o formulário
with st.form("form_avaliacao"):
    
    # Organizando em colunas para ficar visualmente melhor
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dados Cadastrais")
        genero_user = st.radio("Gênero", ["F", "M"])
        idade = st.number_input("Idade", min_value=6, max_value=25, value=14)
        ano_ingresso = st.number_input("Ano de Ingresso", min_value=2010, max_value=2024, value=2022)
        ano_atual = st.number_input("Ano da Avaliação", value=2024)
        escola_user = st.selectbox("Instituição de Ensino", ["Escola Pública", "Escola Privada"])
        turma = st.text_input("Turma (opcional, pode não ser usada no modelo)", value="3A")


    with col2:
        st.subheader("Fases e Notas")
        fase_user = st.selectbox("Fase Atual (Nível de Aprendizagem do Aluno)", list(mapeamento_fases.keys()))
        fase_ideal_user = st.selectbox("Fase Ideal (Nível Ideal do Aluno)", list(mapeamento_fases.keys()))
        pedra_user = st.selectbox("Pedra Associada (Classificação do Aluno baseado no INDE)", list(mapeamento_pedras.keys()))
        qtde_av = st.number_input("Qtde Avaliações do Aluno", min_value=1, value=4)
        matematica = st.number_input("Nota Matemática", min_value=0.0, max_value=10.0, value=6.0)
        portugues = st.number_input("Nota Português", min_value=0.0, max_value=10.0, value=7.0)
        ingles = st.number_input("Nota Inglês", min_value=0.0, max_value=10.0, value=6.5)


    with col3:
        st.subheader("Índices")
        iaa = st.number_input("IAA (Indicador de Auto Avaliação)", min_value=0.0, max_value=10.0, value=7.5, format="%.2f")
        ian = st.number_input("IAN (Indicador de Adequação ao Nível )", min_value=0.0, max_value=10.0, value=6.4, format="%.2f")
        ida = st.number_input("IDA (Indicador de Aprendizagem)", min_value=0.0, max_value=10.0, value=6.8, format="%.2f")
        ieg = st.number_input("IEG (Indicador de Engajamento)", min_value=0.0, max_value=10.0, value=8.1, format="%.2f")
        ips = st.number_input("IPS (Indice Psicosocial)", min_value=0.0, max_value=10.0, value=7.0)
        ipp = st.number_input("IPP (Indice Psicopedagógico)", min_value=0.0, max_value=10.0, value=6.5)
        ipv = st.number_input("IPV (Indice de Ponto de Virada)", min_value=0.0, max_value=10.0, value=7.2)
        inde = st.number_input("INDE (Indice de Desempenho Educacional)", min_value=0.0, max_value=10.0, value=7.1)

    # Botão de Envio do Formulário
    submitted = st.form_submit_button("Rodar Modelo")

# ==========================================
# 3. Processamento e Previsão
# ==========================================
if submitted:
    
    # --- ETAPA CRÍTICA: PRÉ-PROCESSAMENTO ---
    # Convertendo as escolhas de texto em números baseados no mapeamento.
    # Se você tiver os encoders salvos (joblib), substitua essa parte por encoder.transform().
    try:
        fase_mapped = mapeamento_fases[fase_user]
        fase_ideal_mapped = mapeamento_fases[fase_ideal_user]
        pedra_mapped = mapeamento_pedras[pedra_user]
        genero_mapped = mapeamento_genero[genero_user]
        escola_mapped = mapeamento_instituicao[escola_user]
    except KeyError:
        st.error("Erro interno no mapeamento de categorias. Verifique os dicionários manuais.")
        st.stop()

    # Criando o DataFrame de entrada com os valores transformados
    # A ordem das colunas DEVE ser EXATAMENTE a mesma que o modelo foi treinado.
    # Verifique isso com seu parceiro!
    dados_para_prever = pd.DataFrame([{
        "idade": idade,
        "ano_ingresso": ano_ingresso,
        "iaa": iaa,
        "ian": ian,
        "ida": ida,
        "ieg": ieg,
        "ips": ips,
        "ipp": ipp,
        "ipv": ipv,
        "inde": inde,
        "matematica": matematica,
        "portugues": portugues,
        "ingles": ingles,
        "qtde_av": qtde_av,
        "ano": ano_atual,
        "fase": fase_mapped,              # Número
        "fase_ideal": fase_ideal_mapped,  # Número
        "pedra": pedra_mapped,            # Número
        "genero": genero_mapped,          # Número
        "instituicao_ensino": escola_mapped, # Número
        "turma": turma # Geralmente turma não entra no modelo
    }])
    
    # Exibir os dados que vão para o modelo (para debug)
    # st.write("Dados processados enviados ao modelo:", dados_para_prever)

    # --- Executando o Modelo ---
    with st.spinner('Processando previsão...'):
        try:
            # st.cache_resource garante que o modelo já esteja carregado
            
            dados_para_prever = dados_para_prever[features]
            previsao = modelo.predict_proba(dados_para_prever)[0][1]

     
            # --- Exibindo o Resultado ---
            st.subheader("Resultado da Avaliação:")
            
            # Supondo que 'previsao' seja a probabilidade ou valor retornado pelo modelo
            if previsao > 0.6:
                st.error(f"🚨 O resultado é: **RISCO ALTO - Valor: {previsao}**")
                # st.snow() # Opcional: efeito de neve para alertas críticos
                
            elif previsao > 0.3:
                st.warning(f"⚠️ O resultado é: **RISCO MÉDIO - Valor: {previsao}**")
                
            else:
                st.success(f"✅ O resultado é: **RISCO BAIXO - Valor: {previsao}**")
                st.balloons() # Comemoração para risco baixo/sucesso
        
        except Exception as e:
            st.error(f"Erro durante a previsão do modelo. Isso geralmente acontece se a ordem das colunas ou o formato dos dados (texto vs número) estiver incorreto em relação ao treino. Erro: {e}")