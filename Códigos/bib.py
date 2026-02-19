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
    # Função para calcular o vetor de sobrevivência entre x até w - x
    def expec_vida(vetor_qx: np.ndarray, idade_alvo: int):

        """
        Gera um vetor de probabilidades de sobrevivência a partir de um vetor de probabilidades de morte.

        Args:
            vetor_qx (vetor contendo as probabilidades).
            idade_alvo (idade de interesse).

        Returns:
            Vetor de probabilidades de sobrevivência.
        """
        
        # Converte para array
        qx = np.array(vetor_qx)
        
        # Calcula px
        px = 1 - qx
        
        # Calcula lx
        lx = 100000.0 * np.cumprod(np.concatenate([[1], px[:-1]]))
        
        # Calcula t_p_x para t de 0 até ω-x
        t_p_x = lx[idade_alvo:] / lx[idade_alvo]
        
        # Retorna o vetor de probailidades
        return t_p_x
    ###########################################################################################################################################################

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o vetor de desconto financeiros
    def vetor_desconto(v: float, n: int):
        
        """
        Gera um vetor um vetor com descontos financieros.

        Args:
            v: Função de desconto financeiro (já incorporada o juros).
            n: Tamanho do vetor.

        Returns:
            Vetor de vetor de desconto financeiros.
        """
        
        # Calcula um vetor descontos financeiros de tamanho "n"
        vetor = [pow(v, x) for x in range(0, n)]
        
        # Retorna o vetor
        return vetor
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

    ###########################################################################################################################################################
    @staticmethod
    # Função para calcular o fator de comparação entre tábuas
    def fator_K(x: np.array, y: int, r: int, Sy: float, tabua: pd.DataFrame, i: float, IS: float, custeio: str):

        """
        Calcula o fator K que sintetiza o efeito da tábua de mortalidade sobre a provisão matemática,
        permitindo comparações diretas entre tábuas.
        
        Args:
            x (np.array): Idades atuais dos segurados.
            y (int): Idade de entrada no plano (constante).
            r (int): Idade de aposentadoria (constante).
            Sy (float): Salário na idade de entrada y.
            tabua (pd.DataFrame): Tábua de mortalidade com colunas 'Idade' e 'lx'.
            i (float): Taxa real de juros anual (ex: 0.06 para 6% a.a.).
            IS (float): Taxa real de crescimento salarial anual (ex: 0.02 para 2% a.a.).
            custeio (str): Método de custeio ('INE', 'CUP' ou 'PNI').
        
        Returns:
            np.array: Fatores K correspondentes a cada idade em x.
        
        """

        # Criar tabela de comutação
        comutacao = tabua.copy()
        comutacao["Dx"] = comutacao["lx"] * ((1 / (1 + i)) ** comutacao["Idade"])
        comutacao["Nx"] = comutacao["Dx"][::-1].cumsum()[::-1]
        comutacao["Dxs"] = comutacao["Dx"] * ((1 / (1 + IS)) ** comutacao["Idade"])
        comutacao["Nxs"] = comutacao["Dxs"][::-1].cumsum()[::-1]

        # Transformar em dicionários
        dict_Dx = dict(zip(comutacao["Idade"], comutacao["Dx"]))
        dict_Nx = dict(zip(comutacao["Idade"], comutacao["Nx"]))
        dict_Dxs = dict(zip(comutacao["Idade"], comutacao["Dxs"]))
        dict_Nxs = dict(zip(comutacao["Idade"], comutacao["Nxs"]))

        # Obtém os valores de Dx, Nx, Dxs e Nxs
        Nx = np.array([dict_Nx.get(idade, 0) for idade in x])
        Dx = np.array([dict_Dx.get(idade, 0) for idade in x])
        Nxs = np.array([dict_Nxs.get(idade, 0) for idade in x])
        Dxs = np.array([dict_Dxs.get(idade, 0) for idade in x])

        # Obtém os valores para Dy, Ny, Nr, Dys, Nys Nrs (constantes)
        Dy = dict_Dx.get(y, 0)
        Ny = dict_Nx.get(y, 0)
        Nr = dict_Nx.get(r, 0)
        Dys = dict_Dxs.get(y, 0)
        Nys = dict_Nxs.get(y, 0)
        Nrs = dict_Nxs.get(r, 0)

        # Máscara booleana para x < r
        mask_contribuicao = x < r

        # Para o método de custeio INE
        if custeio == "INE":

            # Inicializar array de resultados
            fator_ine = np.zeros_like(x, dtype=float)

            # Calcula o fator K para a PMBaC
            fator_ine[mask_contribuicao] = (Nr / Dx[mask_contribuicao]) * ((Ny - Nx[mask_contribuicao]) / (Ny - Nr))

            # Calcula o fator K para a PMBC
            fator_ine[~mask_contribuicao] = Nx[~mask_contribuicao] / Dx[~mask_contribuicao]

            # Retorna o vetor com os respectivos fatores K 
            return fator_ine
        
        # Para o método de custeio CUP
        elif custeio == "CUP":

            # Inicializar array de resultados
            fator_cup = np.zeros_like(x, dtype=float)

            # Calcula o fator K para a PMBaC
            fator_cup[mask_contribuicao] = (Nr / Dx[mask_contribuicao]) * ((x[mask_contribuicao] - y) / (r - y))

            # Calcula o fator K para a PMBC
            fator_cup[~mask_contribuicao] = Nx[~mask_contribuicao] / Dx[~mask_contribuicao]

            # Retorna o vetor com os respectivos fatores K 
            return fator_cup

        # Para o método de custeio PNI
        elif custeio == "PNI":

            # Inicializar array de resultados
            fator_pni = np.zeros_like(x, dtype=float)

            # Calcula o fator K para a PMBaC
            fator_pni[mask_contribuicao] = (Nr / Dx[mask_contribuicao]) - ((1 + IS) ** (x[mask_contribuicao] - y)) * (Nr / Dy) * (Dys / Dxs[mask_contribuicao]) * ((Nxs[mask_contribuicao] - Nrs) / (Nys - Nrs))
            
            # Calcula o fator K para a PMBC
            fator_pni[~mask_contribuicao] = Nx[~mask_contribuicao] / Dx[~mask_contribuicao]

            # Retorna o vetor com os respectivos fatores K 
            return fator_pni


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

class Bases:

    ###########################################################################################################################################################
    @staticmethod
    # Função para fazer o bootstrapping da base de dados
    def bootstrapping_idade_alvo(base_bruta: pd.DataFrame, tamanho_amostra: int, idade_media_alvo: float, seed: int, max_iteracoes=100000):
        """
        Gera uma amostra aleatória com reposição que atinge uma média de idade específica.

        Args:
            base_bruta : Base de dados original. Deve conter a coluna 'Idade'.
            tamanho_amostra : Número de registros a serem amostrados (permite valores maiores que a base original).
            idade_media_alvo : Média de idade desejada para a amostra final.
            seed: Semente para reprodução.
            max_iteracoes : Número máximo de ajustes para convergir à média alvo (padrão: 10000).
        
        Returns:
            pd.DataFrame: Amostra com índices resetados contendo aproximadamente a média de idade alvo.
        """
        
        # Seed
        np.random.seed(seed)
        
        # Converte coluna 'Idade' para array
        idades = base_bruta['Idade'].values
        indices_base = base_bruta.index.values
        
        # Divide a base em: idades abaixo e acima da mediana
        idade_mediana = np.median(idades)
        indices_baixos = np.where(idades <= idade_mediana)[0]
        indices_altos = np.where(idades > idade_mediana)[0]

        # Amostragem inicial - Calcula probabilidades favorecendo idades próximas ao alvo
        diferencas = np.abs(idades - idade_media_alvo)
        
        # Função exponencial, quanto mais próximo, maior o peso
        pesos = np.exp(-diferencas / 10)
        
        # Normaliza para somar 1
        probabilidades = pesos / pesos.sum()
        
        # Sorteia índices, com reposição, usando as probabilidades calculadas
        indices_selecionados = np.random.choice(len(idades), size=tamanho_amostra, replace=True, p=probabilidades)
        
        # Ajuste fino por substituição direcionada
        idades_amostra = idades[indices_selecionados].copy()
        
        # Aceita diferença de até 0.01 anos
        tolerancia = 0.01 
        
        for iteracao in range(max_iteracoes):
            media_atual = idades_amostra.mean()
            diferenca = idade_media_alvo - media_atual

            # Verifica se já atingiu a média desejada
            if abs(diferenca) < tolerancia:
                break

            # Quanto maior a diferença, mais agressivo o ajuste
            if abs(diferenca) > 5:
                n_substituicoes = min(100, tamanho_amostra // 5)
            elif abs(diferenca) > 1:
                n_substituicoes = min(50, tamanho_amostra // 10)
            else:
                n_substituicoes = min(10, tamanho_amostra // 20)

            # Média atual está abaixo do alvo
            if diferenca > 0:
                # Substitui as N menores idades por idades maiores
                idx_menores = np.argpartition(idades_amostra, n_substituicoes)[:n_substituicoes]
                novos_idx = np.random.choice(indices_altos, size=n_substituicoes, replace=True)

            # Média atual está acima do alvo
            else:
                # Substitui as N maiores idades por idades menores
                idx_maiores = np.argpartition(idades_amostra, -n_substituicoes)[-n_substituicoes:]
                novos_idx = np.random.choice(indices_baixos, size=n_substituicoes, replace=True)
                idx_menores = idx_maiores

            # Aplica as substituições
            indices_selecionados[idx_menores] = novos_idx
            idades_amostra[idx_menores] = idades[novos_idx]

        # Debug: imprime se não convergiu
        media_final = idades_amostra.mean()

        # Converte de volta para DataFrame
        amostra_final = base_bruta.iloc[indices_selecionados].copy()
        amostra_final = amostra_final.reset_index(drop=True)
        
        return amostra_final 
    ###########################################################################################################################################################

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
                idades_simuladas.extend(np.random.randint(19, 25, size=num_pessoas_faixa))
            elif faixa == "26-30":
                idades_simuladas.extend(np.random.randint(26, 30, size=num_pessoas_faixa))
            elif faixa == "31-35":
                idades_simuladas.extend(np.random.randint(31, 35, size=num_pessoas_faixa))
            elif faixa == "36-40":
                idades_simuladas.extend(np.random.randint(36, 40, size=num_pessoas_faixa))
            elif faixa == "41-45":
                idades_simuladas.extend(np.random.randint(41, 45, size=num_pessoas_faixa))
            elif faixa == "46-50":
                idades_simuladas.extend(np.random.randint(46, 50, size=num_pessoas_faixa))
            elif faixa == "51-55":
                idades_simuladas.extend(np.random.randint(51, 55, size=num_pessoas_faixa))
            elif faixa == "56-60":
                idades_simuladas.extend(np.random.randint(56, 60, size=num_pessoas_faixa))
            elif faixa == "61-65":
                idades_simuladas.extend(np.random.randint(61, 65, size=num_pessoas_faixa))
            elif faixa == "66-70":
                idades_simuladas.extend(np.random.randint(66, 70, size=num_pessoas_faixa))
            elif faixa == "71-75":
                idades_simuladas.extend(np.random.randint(71, 75, size=num_pessoas_faixa))
            elif faixa == "76-80":
                idades_simuladas.extend(np.random.randint(76, 80, size=num_pessoas_faixa))
            elif faixa == "81-85":
                idades_simuladas.extend(np.random.randint(81, 85, size=num_pessoas_faixa))
            elif faixa == "86-90":
                idades_simuladas.extend(np.random.randint(86, 90, size=num_pessoas_faixa))
            elif faixa == "91-95":
                idades_simuladas.extend(np.random.randint(91, 95, size=num_pessoas_faixa))
            elif faixa == "96-100":
                idades_simuladas.extend(np.random.randint(96, 100, size=num_pessoas_faixa))

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