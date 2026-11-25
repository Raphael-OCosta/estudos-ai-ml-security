"""Dados fictícios do laboratório de segurança de IA da Fortaleza Digital, gerados com semente fixa.

Use nos cadernos com `from dados import *` (a Preparação já faz isso).

  digitos()            os dígitos 8x8 do scikit-learn (vêm junto com a biblioteca), escalados para [0, 1]
  fluxos_rede()        fluxos de uma rede industrial (IIoT) com 4 tipos de ataque, como os conjuntos NSL-KDD/Edge-IIoT
  interesses_ndn()     taxa de "interesses" por roteador ao longo do tempo, com ataques de inundação (IFA)
  transacoes_banco()   transações de internet banking com fraudes
  emails()             e-mails rotulados como phishing ou legítimos
  REDE_IOT             a estrutura e as probabilidades de uma rede bayesiana de ameaças a IoT
"""
import numpy as np
import pandas as pd


def digitos():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data / 16.0, d.target


def fluxos_rede(n=12000, semente=5):
    r = np.random.default_rng(semente)
    tipo = r.choice(["normal", "dos", "varredura", "força bruta", "exfiltração"], n, p=[.72, .12, .08, .05, .03])
    base = {"duracao_s": (2.0, 1.5), "bytes_env": (6.0, 1.0), "bytes_rec": (7.5, 1.2), "pacotes": (3.0, .8),
            "portas_distintas": (0.3, .5), "falhas_login": (0.05, .2), "taxa_syn": (0.1, .08), "hora": (13, 4)}
    df = pd.DataFrame({k: r.normal(m, s, n) for k, (m, s) in base.items()})
    e = lambda t: tipo == t
    df.loc[e("dos"), "pacotes"] += r.normal(3.0, .6, e("dos").sum()); df.loc[e("dos"), "taxa_syn"] += r.normal(.6, .15, e("dos").sum())
    df.loc[e("dos"), "duracao_s"] -= 1.5
    df.loc[e("varredura"), "portas_distintas"] += r.normal(3.5, 1.0, e("varredura").sum()); df.loc[e("varredura"), "bytes_rec"] -= 3
    df.loc[e("força bruta"), "falhas_login"] += r.normal(2.5, .8, e("força bruta").sum()); df.loc[e("força bruta"), "hora"] = r.normal(3, 2, e("força bruta").sum())
    df.loc[e("exfiltração"), "bytes_env"] += r.normal(3.5, .9, e("exfiltração").sum()); df.loc[e("exfiltração"), "hora"] = r.normal(2, 2, e("exfiltração").sum())
    df = df.clip(lower=0).round(3)
    df["hora"] = (df["hora"] % 24).round(1)
    df["protocolo"] = r.choice(["tcp", "udp", "modbus", "mqtt"], n, p=[.5, .2, .15, .15])
    df["tipo"] = tipo
    df["ataque"] = (tipo != "normal").astype(int)
    return df


def interesses_ndn(roteadores=20, passos=300, semente=6):
    """Interesses por segundo e taxa de interesses pendentes não atendidos (PIT) por roteador; ataque nos passos 150-220."""
    r = np.random.default_rng(semente)
    atacados = {3, 7, 11, 15, 18}          # dois deles caem no teste (roteadores 14 a 19)
    linhas = []
    for rt in range(roteadores):
        base = r.uniform(80, 150)
        for t in range(passos):
            ataque = rt in atacados and 150 <= t < 220
            # o ataque é discreto: soma algo comparável à variação normal de um roteador movimentado
            taxa = base * (1 + 0.15 * np.sin(t / 15)) + r.normal(0, 12) + (r.uniform(20, 70) if ataque else 0)
            pit = r.beta(2, 18) + (r.uniform(.03, .15) if ataque else 0)
            linhas.append((rt, t, round(taxa, 1), round(min(pit, 1), 3), int(ataque)))
    return pd.DataFrame(linhas, columns=["roteador", "t", "interesses_s", "pit_nao_atendido", "ataque"])


def transacoes_banco(n=20000, semente=7):
    r = np.random.default_rng(semente)
    fraude = r.random(n) < 0.02
    df = pd.DataFrame({
        "valor": np.where(fraude, r.lognormal(7.2, .6, n), r.lognormal(5.0, 1.0, n)),
        "hora": np.where(fraude, r.normal(3, 2.5, n), r.normal(14, 4, n)) % 24,
        "dispositivo_novo": np.where(fraude, r.random(n) < .7, r.random(n) < .05).astype(int),
        "distancia_km": np.where(fraude, r.lognormal(5, 1, n), r.lognormal(1.5, 1, n)),
        "tentativas_senha": np.where(fraude, r.poisson(2.2, n), r.poisson(.15, n)),
        "transacoes_24h": np.where(fraude, r.poisson(7, n), r.poisson(2, n)),
    }).round(2)
    df["fraude"] = fraude.astype(int)
    return df


_NEUTRAS = ("relatório reunião equipe projeto prazo cliente contrato planilha aprovação entrega apresentação diretor "
           "horas revisão orçamento fornecedor pedido nota fiscal agenda amanhã sexta semana mês favor segue anexo "
           "obrigado atenciosamente prezado conforme combinado dados cadastro acesso pagamento fatura").split()
GATILHOS = ["senha", "clique", "urgente", "link", "conta", "bloqueada", "cartão", "token", "prêmio", "verificar",
            "atualize", "imediatamente"]


def emails(n=3000, semente=8):
    """Legítimos usam só palavras de trabalho (às vezes 'link' ou 'conta' de forma inocente); phishing mistura gatilhos.

    Cerca de 20% dos legítimos vêm de parceiros externos e 25% dos phishing imitam o domínio interno."""
    r = np.random.default_rng(semente)
    linhas = []
    for _ in range(n):
        phishing = r.random() < .3
        palavras = list(r.choice(_NEUTRAS, int(r.integers(4, 9))))
        if phishing:
            palavras += list(r.choice(GATILHOS, int(r.integers(2, 5)), replace=False))
        elif r.random() < .15:
            palavras.append(str(r.choice(["link", "conta"])))
        r.shuffle(palavras)
        if phishing:
            remetente = ("ti@fortaleza-digital.com.br" if r.random() < .25 else
                         str(r.choice(["suporte@banco-seguro.co", "premios@promo-br.net", "ti@fortaleza-digital.com.br.verifica.io"])))
            links = int(r.integers(0, 4))
        else:
            remetente = str(r.choice(["joao@fornecedor.com.br", "contato@parceiro.com"])) if r.random() < .2 else \
                str(r.choice(["ana@fortaleza-digital.com.br", "rh@fortaleza-digital.com.br", "bruno@fortaleza-digital.com.br"]))
            links = int(r.random() < .25)
        linhas.append((remetente, " ".join(palavras).capitalize(), links, int(phishing)))
    return pd.DataFrame(linhas, columns=["remetente", "texto", "n_links", "phishing"])


# Rede bayesiana de ameaças a um dispositivo IoT (cap. 7). Nós binários (1 = verdadeiro).
REDE_IOT = {
    "senha_padrao": {"pais": [], "p": 0.30},
    "firmware_antigo": {"pais": [], "p": 0.40},
    "porta_exposta": {"pais": [], "p": 0.25},
    "invasao": {"pais": ["senha_padrao", "firmware_antigo", "porta_exposta"],
                "p": {(0, 0, 0): 0.01, (0, 0, 1): 0.10, (0, 1, 0): 0.08, (0, 1, 1): 0.35,
                      (1, 0, 0): 0.15, (1, 0, 1): 0.55, (1, 1, 0): 0.40, (1, 1, 1): 0.85}},
    "trafego_anomalo": {"pais": ["invasao"], "p": {(0,): 0.05, (1,): 0.80}},
    "alerta_ids": {"pais": ["trafego_anomalo"], "p": {(0,): 0.02, (1,): 0.90}},
    "vazamento": {"pais": ["invasao"], "p": {(0,): 0.001, (1,): 0.30}},
}
