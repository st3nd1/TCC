# Bibliotecas
import pandas as pd
import numpy as np
from datetime import datetime

class Calculos:

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular a tabua de vida
    def calcular_tabua(vetor_qx: np.ndarray):

        """
        Gera uma tábua de mortalidade completa a partir de um vetor de probabilidades de morte.

        Args:
            vetor_qx (vetor contendo as probabilidades).

        Returns:
            tábua de mortalidade em um pandas dataframe.
        """
        
        # Cria DataFrame com dados iniciais
        tabua = pd.DataFrame({"Idade": np.arange(len(vetor_qx)), "qx": vetor_qx})

        # Calcula px
        tabua["px"] = 1 - tabua["qx"]

        # Calcula lx
        tabua["lx"] = 100000.0 * np.concatenate([[1], tabua["px"].iloc[:-1].cumprod().values])

        # Calcula dx
        tabua["dx"] = tabua["lx"] * tabua["qx"]

        # Calcula Lx
        tabua["Lx"] = tabua["lx"] - (0.5 * tabua["dx"])

        # Calcula Tx
        tabua["Tx"] = tabua["Lx"][::-1].cumsum()[::-1]

        # Calcula ex
        tabua["ex"] = tabua["Tx"] / tabua["lx"]

        # Retorna a tábua calculada
        return tabua
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o VABF do PMBaC
    def VABF_PMBaC(base_cadastral: pd.DataFrame, tabua: pd.DataFrame, i: float, IS: float):
        
        """
        Calcula o Valor Atual dos Benefícios Futuros (VABF) referente a provisão matemática de benefícios a conceder PMBaC.
        
        Args:
            base_cadastral (pd.DataFrame): Dados dos participantes.
            tabua (pd.DataFrame): Tábua de mortalidade.
            i (float): Taxa de juros real anual
            IS (float): Fator de crescimento salarial.

        Returns:
            np.ndarray: O valor referente ao VABF para o PMBaC.
        """
        
        # Reseta os índices dos DataFrames
        tabua = tabua.reset_index(drop=True)
        base_cadastral = base_cadastral.reset_index(drop=True)

        # Copia a tábua de vida para calcular a tabela de comutação
        comutacao = tabua.copy()

        # Calcula as funções de comutação Dx e Nx
        comutacao["Dx"] = comutacao["lx"] * ((1 / (1 + i)) ** comutacao["Idade"])
        comutacao["Nx"] = comutacao["Dx"][::-1].cumsum()[::-1]

        # Cria dicionários das funções de comutação para acesso rápido
        dict_Dx = dict(zip(comutacao["Idade"], comutacao["Dx"]))
        dict_Nx = dict(zip(comutacao["Idade"], comutacao["Nx"]))

        # Vetor para as idades e tempos de contribuições
        x = base_cadastral["Idade"].values
        r = base_cadastral["Idade Provável de Aposentadoria"].values
        y = base_cadastral["Idade ingresso"].values
        tempo_total_contribuicao =  r - y
        
        # Vetor para salario na idade y (entrada)
        s = base_cadastral["Remuneração"].values

        # Vetor para o benificio futuro na idade r (aposentadoria)
        b = (s * (1+IS) ** tempo_total_contribuicao) * 0.8

        # Vetores para cada Dx e Nr
        Dx = np.array([dict_Dx[idade] for idade in x])
        Nr = np.array([dict_Nx[idade] for idade in r])
        
        # Vetor para a anuidade vitalicia diferida
        m_a_x = np.where(Dx != 0, Nr / Dx, 0)

        # Vetor do VABF individual
        vabf_individual = m_a_x * b

        # Retorna o vetor do VABF
        return vabf_individual
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o VABF do PMBC
    def VABF_PMBC(base_cadastral: pd.DataFrame, tabua: pd.DataFrame, taxa: float, IS: float):
        
        """
        Calcula o Valor Atual dos Benefícios Futuros (VABF) para a Provisão Matemática de Beneficios Concedidos (PMBC).
        
        Args:
            base_cadastral (pd.DataFrame): Dados dos participantes.
            tabua (pd.DataFrame): Tábua de mortalidade.
            taxa (float): Taxa de juros real anual esperada.
            IS (float): Taxa de crescimento salarial.

        Returns:
            np.ndarray: O valor referente ao VABF para a PMBC.
        """
            
        # Reseta os índices dos DataFrames
        tabua = tabua.reset_index(drop=True)
        base_cadastral = base_cadastral.reset_index(drop=True)

        # Cria tabela de comutação
        comutacao = tabua.copy()

        # Calcula as funções de comutação Dx e Nx
        comutacao["Dx"] = comutacao["lx"] * ((1 / (1 + taxa)) ** comutacao["Idade"])
        comutacao["Nx"] = comutacao["Dx"][::-1].cumsum()[::-1]

        # Calcula a renda vitalicia com comutação
        comutacao["ax"] = np.where(comutacao["Dx"] != 0, comutacao["Nx"] / comutacao["Dx"], 0)

        # Cria dicionários das funções da renda vitalicia para acesso rápido
        dict_ax = dict(zip(comutacao["Idade"], comutacao["ax"]))

        # Vetor para o beneficio e a idade atual para todos os participantes
        x = base_cadastral["Idade"].values
        r = base_cadastral["Idade Provável de Aposentadoria"].values
        y = base_cadastral["Idade ingresso"].values
        tempo_total_contribuicao =  r - y

        # Vetor para salario na idade y (entrada)
        s = base_cadastral["Remuneração"].values

        # Vetor para o benificio futuro na idade r (aposentadoria)
        b = (s * (1+IS) ** tempo_total_contribuicao) * 0.8

        # Vetor com a renda vitalicia correspondente à idade atual
        ax = np.array([dict_ax[i] for i in x])

        # Vetor com o VABF individual
        vabf_individual = ax * b

        # Retorna o vetor do VABF
        return vabf_individual
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o VACF do PMBaC por IEN
    def VACF_PMBaC_INE(base_cadastral: pd.DataFrame, tabua: pd.DataFrame, VABFx: np.ndarray, i: float):
        """
        Calcula o Valor Atual das Contribuições Futuras (VACF) para a Provisão Matemática de Beneficios a Conceder (PMBaC) usando
        o método de custeio Idade Normal de Entrada (IEN).

        Args:
            base_cadastral (pd.DataFrame): Base de dados dos ativos.
            tabua (pd.DataFrame): Tábua de mortalidade.
            VABFx (np.ndarray): Valor Atual dos Benefícios Futuros na idade x.
            i (float): Taxa de juros real anual.


        Returns:
            np.ndarray: O valor referente ao VACF para a PMBaC.
        """

        # Reseta os índices dos DataFrames
        tabua = tabua.reset_index(drop=True)
        base_cadastral = base_cadastral.reset_index(drop=True)

        # Copia a tábua de vida para calcular a tabela de comutação
        comutacao = tabua.copy()
        
        # Calcula as funções de comutação Dx e Nx
        comutacao["Dx"] = comutacao["lx"] * ((1 / (1 + i)) ** comutacao["Idade"])
        comutacao["Nx"] = comutacao["Dx"][::-1].cumsum()[::-1]

        # Cria dicionários das funções de comutação para acesso rápido
        dict_Dx = dict(zip(comutacao["Idade"], comutacao["Dx"]))
        dict_Nx = dict(zip(comutacao["Idade"], comutacao["Nx"]))

        # Vetor para o salário e para as idades
        s = base_cadastral["Remuneração"].values
        x = base_cadastral["Idade"].values
        r = base_cadastral["Idade Provável de Aposentadoria"].values
        y = base_cadastral["Idade ingresso"].values
        
        # Vetor com os valores de Nx nas idades r (aposentadoria)
        Nr = np.array([dict_Nx.get(idade, 0) for idade in r])
        
        # Vetor com os valores de Dx e Nx na idade x (atual)
        Dx = np.array([dict_Dx.get(i, 0) for i in x])
        Nx = np.array([dict_Nx.get(i, 0) for i in x])

        # Vetor com os valores de Dx e Nx na idade y (ingresso)
        Dy = np.array([dict_Dx.get(i, 0) for i in y])
        Ny = np.array([dict_Nx.get(i, 0) for i in y])

        # Vetor com os valores das anuidades temporárias (x até r, y até r)
        axn = np.where(Dx != 0, (Nx - Nr) / Dx, 0)
        ayn = np.where(Dy != 0, (Ny - Nr) / Dy, 0)

        # Fator de capitalização atuarial
        Exy = np.where(Dy != 0, Dx / Dy, 0)
        
        # Vetor do VABF na idade y (ingresso)
        VABFy = Exy * VABFx

        # Vetor do Custo Normal
        CN = np.where(ayn != 0, VABFy / ayn, 0)

        # Vetor com o VACF individual
        vacf_individual = axn * CN

        # Retorna o vetor com VACF
        return vacf_individual
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o VACF do PMBaC por CUP
    def VACF_PMBaC_CUP(base_cadastral: pd.DataFrame, VABFx: np.ndarray):
        """
        Calcula o Valor Atual das Contribuições Futuras (VACF) para a Provisão Matemática de Beneficios a Conceder (PMBaC) usando 
        método de custeio Crédito Único Projetado (CUP).

        Args:
            base_cadastral (pd.DataFrame): Base de dados dos ativos.
            VABFx (np.ndarray): Valor Atual dos Benefícios Futuros na idade x.

        Returns:
            np.ndarray: O valor referente ao VACF para a PMBaC.
        """
        
        # Reseta os índices do DataFrame
        base_cadastral = base_cadastral.reset_index(drop=True)

        # Vetor para as idades e tempos de contribuições
        x = base_cadastral["Idade"].values
        r = base_cadastral["Idade Provável de Aposentadoria"].values
        y = base_cadastral["Idade ingresso"].values
        tempo_total_contribuicao = r - y
        tempo_futuro_contribuicao = r - x

        # Vetor para o custo normal
        CN = VABFx / tempo_total_contribuicao
        
        # Vetor do VACF individual
        vacf_individual = CN * tempo_futuro_contribuicao
        
        # Retorna o vetor do VACF
        return vacf_individual
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o VACF do PMBaC por PNI
    def VACF_PMBaC_PNI(base_cadastral: pd.DataFrame, tabua: pd.DataFrame, VABFx: np.ndarray, i: float, IS: float):
        """
        Calcula o Valor Atual das Contribuições Futuras (VACF) para a Provisão Matemática de Beneficios a Conceder (PMBaC) usando 
        método de custeio Prêmio Nivelado Individual (PNI).

        Args:
            base_cadastral (pd.DataFrame): Base de dados dos ativos.
            tabua (pd.DataFrame): Tábua de mortalidade.
            VABFx (np.ndarray): Valor Atual dos Benefícios Futuros na idade x.
            i (float): Taxa de juros real anual
            IS (float): Fator de crescimento salarial.

        Returns:
            np.ndarray: O valor referente ao VACF para a PMBaC.
        """

        # Reseta os índices dos DataFrames
        tabua = tabua.reset_index(drop=True)
        base_cadastral = base_cadastral.reset_index(drop=True)

        # Copia a tábua de vida para calcular a tabela de comutação
        comutacao = tabua.copy()

        # Calcula as funções de comutação Dx e Nx
        comutacao["Dx"] = comutacao["lx"] * ((1 / (1 + i)) ** comutacao["Idade"])
        comutacao["Nx"] = comutacao["Dx"][::-1].cumsum()[::-1]

        # Calcula as funções de comutação Dx e Nx com crescimento salarial
        comutacao["Dxs"] = comutacao["Dx"] * ((1 / (1 + IS)) ** comutacao["Idade"])
        comutacao["Nxs"] = comutacao["Dxs"][::-1].cumsum()[::-1]

        # Cria dicionários das funções de comutação para acesso rápido
        dict_Dx = dict(zip(comutacao["Idade"], comutacao["Dx"]))
        dict_Nx = dict(zip(comutacao["Idade"], comutacao["Nx"]))
        dict_Dxs = dict(zip(comutacao["Idade"], comutacao["Dxs"]))
        dict_Nxs = dict(zip(comutacao["Idade"], comutacao["Nxs"]))
        
        # Vetor para as idades e tempos de contribuições
        x = base_cadastral["Idade"].values
        r = base_cadastral["Idade Provável de Aposentadoria"].values
        y = base_cadastral["Idade ingresso"].values
        tempo_passado_contribuicao = x - y
        tempo_total_contribuicao = r - y
        tempo_futuro_contribuicao = r - x

        # Vetor para os salários nas idades x (atual), r (aposentadoria) e y (ingresso)
        sx = base_cadastral["Remuneração"].values
        sr = sx * (1 + IS) ** tempo_futuro_contribuicao
        sy = sx * (1 + IS) ** (-tempo_passado_contribuicao)

        # Vetores para cada Dx nas idades x (atual) e y (ingresso) 
        Dx = np.array([dict_Dx.get(i, 0) for i in x])
        Dy = np.array([dict_Dx.get(i, 0) for i in y])
        
        # Vetor para cada Nx na idades r (aposentadoria) com crescimento salarial
        Nrs = np.array([dict_Nxs.get(idade, 0) for idade in r])
        
        # Vetores para cada Dx e Nx na idade x (atual) com crescimento salarial
        Dxs = np.array([dict_Dxs.get(i, 0) for i in x])
        Nxs = np.array([dict_Nxs.get(i, 0) for i in x])

        # Vetores para cada Dx e Nx na idade y (ingresso) com crescimento salarial
        Dys = np.array([dict_Dxs.get(i, 0) for i in y])
        Nys = np.array([dict_Nxs.get(i, 0) for i in y])

        # Vetores das anuidades temporárias com crescimento salarial
        axns = np.where(Dx != 0, (Nxs - Nrs) / Dxs, 0)
        ayns = np.where(Dy != 0, (Nys - Nrs) / Dys, 0)

        # Vetor para o fator de capitalização atuarial
        Exy = np.where(Dy != 0, Dx / Dy, 0)

        # Vetor para o Valor Atual dos Salário Futuros nas idade y e x 
        VASFy = sy * axns
        VASFx = sx * ayns

        # Vetor para o Valor Atual dos Benefícios Futuros na idade y
        VABFy = VABFx * Exy

        # Vetor para o Custo Normal
        ACN = VABFy / VASFy

        # Vetor para o VACF individual
        vacf_individual = ACN * VASFx
        
        # Retorna o vetor do VACF
        return vacf_individual
    ###########################################################################################################################################################

class Auxiliares:

    ###########################################################################################################################################################
    @staticmethod
    # Função auxiliar para formatação monetária nas tabelas
    def formatar_brasileiro(valor):
        """
        Formata um valor float para R$

        Args:
            valor (float): valor que se deseja formatar.

        Returns:
            Valor monitário formatado em R$.
        """
        return f"{valor:,.2f}".replace(",", "v").replace(".", ",").replace("v", ".")
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função auxiliar para gráficos
    def formatar_milhar_abs(x, pos):
        """
        Formata um número inteiro para o sistema brasileiro

        Args:
            x (int): valor que se deseja formatar.
        
        Returns:
            Valor formatado com . na casa da milhar.
        """
        return f'{abs(int(x)):,}'.replace(',', '.')
    ###########################################################################################################################################################


class Bases:

    ###########################################################################################################################################################
    @staticmethod
    # Função para gerar idades com base nas proporções
    def gerar_idades(proporcoes, num_pessoas):
        
        """
        Gera uma massa de idades sintética seguindo uma distribuição de frequências por faixas.

        Args:
            proporcoes (dict): Dicionário onde as chaves são as strings das faixas etárias  e os valores são as proporções.
            num_pessoas (int): O tamanho total da amostra populacional a ser gerada.

        Returns:
            np.ndarray: Um array unidimensional contendo as idades simuladas, embaralhadas aleatoriamente.
        """
        
        # Lista para armazenar
        idades_simuladas = []
        
        # Contador
        total_gerado = 0

        # Loop para gerar as idades em cada faixa etária
        for faixa, proporcao in proporcoes.items():
            num_pessoas_faixa = int(num_pessoas * proporcao)
            total_gerado += num_pessoas_faixa

            if faixa == "18-25":
                idades_simuladas.extend(np.random.randint(19, 26, size=num_pessoas_faixa))
            elif faixa == "26-30":
                idades_simuladas.extend(np.random.randint(26, 31, size=num_pessoas_faixa))
            elif faixa == "31-35":
                idades_simuladas.extend(np.random.randint(31, 36, size=num_pessoas_faixa))
            elif faixa == "36-40":
                idades_simuladas.extend(np.random.randint(36, 41, size=num_pessoas_faixa))
            elif faixa == "41-45":
                idades_simuladas.extend(np.random.randint(41, 46, size=num_pessoas_faixa))
            elif faixa == "46-50":
                idades_simuladas.extend(np.random.randint(46, 51, size=num_pessoas_faixa))
            elif faixa == "51-55":
                idades_simuladas.extend(np.random.randint(51, 56, size=num_pessoas_faixa))
            elif faixa == "56-60":
                idades_simuladas.extend(np.random.randint(56, 61, size=num_pessoas_faixa))
            elif faixa == "61-65":
                idades_simuladas.extend(np.random.randint(61, 66, size=num_pessoas_faixa))
            elif faixa == "66-70":
                idades_simuladas.extend(np.random.randint(66, 71, size=num_pessoas_faixa))
            elif faixa == "71-75":
                idades_simuladas.extend(np.random.randint(71, 76, size=num_pessoas_faixa))
            elif faixa == "76-80":
                idades_simuladas.extend(np.random.randint(76, 81, size=num_pessoas_faixa))
            elif faixa == "81-85":
                idades_simuladas.extend(np.random.randint(81, 86, size=num_pessoas_faixa))
            elif faixa == "86-90":
                idades_simuladas.extend(np.random.randint(86, 91, size=num_pessoas_faixa))
            elif faixa == "91-95":
                idades_simuladas.extend(np.random.randint(91, 96, size=num_pessoas_faixa))
            elif faixa == "96-100":
                idades_simuladas.extend(np.random.randint(96, 101, size=num_pessoas_faixa))

        # Converter para array e embaralhar
        idades_array = np.array(idades_simuladas)
        np.random.shuffle(idades_array)

        # Retorna um array com as idades
        return idades_array
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    def base_cadastral(idades_simuladas):
    
        """
        Gera dados aleatórios para criar uma base cadastral completa.

        Args:
            idades_simuladas (DataFrame): DataFrame inicial que deve conter idades e genero.

        Returns:
            pd.DataFrame: Base cadastral processada.
        """
        
        # Obtém ano atual
        ano_atual = datetime.now().year

        # Cria o DataFrame para simulação
        base_simulada = idades_simuladas.copy()

        # Colunas de ID
        base_simulada["Identificação"] = range(1, len(base_simulada) + 1)

        # Coluna de remuneração
        base_simulada["Remuneração"] = 5000

        # Coluna de de idade provável de aposentadoria com base no sexo
        base_simulada["Idade Provável de Aposentadoria"] = np.where(base_simulada["Gênero"] == "Masculino", 65, 62)

        # Coluna de tempo futuro de serviço
        base_simulada['Tempo de serviço futuro'] = base_simulada['Idade Provável de Aposentadoria'] - base_simulada['Idade']

        # Coluna para Idade de ingresso no plano
        base_simulada["Idade ingresso"] = 18

        # Reordenar colunas
        base_simulada = base_simulada[["Identificação", "Gênero", "Remuneração", "Idade", "Idade ingresso", "Idade Provável de Aposentadoria", "Tempo de serviço futuro"]]

        # Retorna o DataFrame
        return base_simulada
    ###########################################################################################################################################################