TITULO = "Modelagem de ameaças a IoT com redes bayesianas"
MISSAO = (
    "Os sensores industriais da Fortaleza Digital podem ser invadidos de vários jeitos: senha de fábrica, firmware "
    "velho, porta exposta. O cap. 7 do livro modela isso com uma **rede bayesiana**: um grafo de causas e efeitos "
    "com probabilidades. Você vai:\n"
    "- implementar a **inferência por enumeração** do zero;\n"
    "- responder perguntas de diagnóstico (\"o IDS alertou, qual a chance de invasão?\");\n"
    "- descobrir qual medida de segurança reduz mais o risco."
)
ENTREGA = "o módulo `PKG/bayes_ameacas.py` (probabilidade conjunta, inferência por enumeração e simulação) e o ranking das medidas de mitigação."
LEITURA = [("7.4 Cyber Security In IoT Networks", 119), ("7.5 Modeling With Bayesian Networks", 121), ("7.6 Model Implementation", 125)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "Uma rede bayesiana fatora a probabilidade conjunta: `P(todos) = Π P(nó | pais)`. Para responder `P(invasão | "
     "alerta = 1)`, some a conjunta sobre todas as combinações das variáveis não observadas (**enumeração**) e "
     "normalize.\n\n"
     "Com 7 nós binários são só 128 combinações, o que dá para fazer na força bruta. Redes grandes usam algoritmos "
     "mais espertos (eliminação de variáveis, amostragem)."),

    ("arquivo",
     "**A inferência.** Crie `bayes_ameacas.py` com:\n"
     "- `prob_no(rede, no, valor, atrib)`: `P(no = valor | pais)`. Para nós sem pais, `p` é um número; com pais, "
     "`p` é um dicionário indexado pela tupla dos valores dos pais;\n"
     "- `conjunta(rede, atrib)`: o produto sobre todos os nós;\n"
     "- `consultar(rede, alvo, evidencia)`: `P(alvo = 1 | evidência)`, por enumeração de todas as atribuições "
     "consistentes com a evidência;\n"
     "- `simular(rede, n, semente)`: amostras na ordem topológica, devolvidas como DataFrame.",
     "bayes_ameacas.py",
     r'''"""Rede bayesiana de ameaças: inferência exata por enumeração e simulação."""
from itertools import product

import numpy as np
import pandas as pd


def prob_no(rede, no, valor, atrib):
    info = rede[no]
    p1 = info["p"] if not info["pais"] else info["p"][tuple(atrib[pai] for pai in info["pais"])]
    return p1 if valor == 1 else 1 - p1


def conjunta(rede, atrib):
    return float(np.prod([prob_no(rede, no, atrib[no], atrib) for no in rede]))


def consultar(rede, alvo, evidencia):
    livres = [n for n in rede if n not in evidencia]
    total = {0: 0.0, 1: 0.0}
    for valores in product([0, 1], repeat=len(livres)):
        atrib = {**evidencia, **dict(zip(livres, valores))}
        total[atrib[alvo]] += conjunta(rede, atrib)
    return total[1] / (total[0] + total[1])


def _ordem(rede):
    feitos, ordem = set(), []
    while len(ordem) < len(rede):
        for n, info in rede.items():
            if n not in feitos and all(p in feitos for p in info["pais"]):
                ordem.append(n)
                feitos.add(n)
    return ordem


def simular(rede, n=10000, semente=0):
    r = np.random.default_rng(semente)
    linhas = []
    for _ in range(n):
        atrib = {}
        for no in _ordem(rede):
            atrib[no] = int(r.random() < prob_no(rede, no, 1, atrib))
        linhas.append(atrib)
    return pd.DataFrame(linhas)'''),

    ("tarefa",
     "**A rede.** Mostre os nós de `REDE_IOT` com os pais de cada um e desenhe o grafo com Matplotlib (nós em "
     "posições fixas e setas dos pais para os filhos). Qual a probabilidade **a priori** de invasão, sem nenhuma "
     "evidência?",
     r'''import matplotlib.pyplot as plt
from PKG.bayes_ameacas import prob_no, conjunta, consultar, simular
for no, info in REDE_IOT.items():
    print(f"{no:<16} pais: {info['pais']}")
pos = {"senha_padrao": (0, 2), "firmware_antigo": (1, 2), "porta_exposta": (2, 2), "invasao": (1, 1),
       "trafego_anomalo": (0.5, 0), "vazamento": (1.8, 0), "alerta_ids": (0.5, -1)}
for no, info in REDE_IOT.items():
    for pai in info["pais"]:
        plt.annotate("", pos[no], pos[pai], arrowprops=dict(arrowstyle="->"))
for no, (x, y_) in pos.items():
    plt.text(x, y_, no, ha="center", bbox=dict(boxstyle="round", fc="#e8f5ee"))
plt.xlim(-0.6, 2.6); plt.ylim(-1.5, 2.5); plt.axis("off"); plt.show()
print("P(invasão) a priori:", round(consultar(REDE_IOT, "invasao", {}), 4))'''),

    ("tarefa",
     "**Diagnóstico.** Calcule:\n"
     "- `P(invasão | alerta_ids = 1)`;\n"
     "- `P(invasão | alerta_ids = 1, senha_padrao = 0)`;\n"
     "- `P(senha_padrao = 1 | invasão = 1)`, o raciocínio \"para trás\";\n"
     "- `P(vazamento | alerta_ids = 1)`.\n\n"
     "Confira o primeiro valor com a **simulação** (100 mil amostras, filtrando as que têm alerta).",
     r'''print("P(invasão | alerta)              ", round(consultar(REDE_IOT, "invasao", {"alerta_ids": 1}), 4))
print("P(invasão | alerta, senha trocada)", round(consultar(REDE_IOT, "invasao", {"alerta_ids": 1, "senha_padrao": 0}), 4))
print("P(senha padrão | invasão)         ", round(consultar(REDE_IOT, "senha_padrao", {"invasao": 1}), 4))
print("P(vazamento | alerta)             ", round(consultar(REDE_IOT, "vazamento", {"alerta_ids": 1}), 4))
sim = simular(REDE_IOT, 100000)
print("simulação P(invasão | alerta):", round(sim.loc[sim.alerta_ids == 1, "invasao"].mean(), 4))'''),

    ("tarefa",
     "**Qual medida vale mais?** Para cada causa (senha, firmware, porta), simule a **mitigação**: copie a rede e "
     "zere a probabilidade da causa (`p = 0`). Calcule a nova `P(vazamento)` e monte o ranking da redução de risco. "
     "Depois combine as duas melhores.",
     r'''import copy
base = consultar(REDE_IOT, "vazamento", {})
efeito = {}
for causa in ["senha_padrao", "firmware_antigo", "porta_exposta"]:
    rede = copy.deepcopy(REDE_IOT); rede[causa]["p"] = 0.0
    efeito[causa] = consultar(rede, "vazamento", {})
print(f"P(vazamento) hoje: {base:.4f}")
for c, v in sorted(efeito.items(), key=lambda kv: kv[1]):
    print(f"corrigir {c:<16}-> {v:.4f} ({1 - v / base:.0%} menos risco)")
duas = sorted(efeito, key=efeito.get)[:2]; rede = copy.deepcopy(REDE_IOT)
for c in duas:
    rede[c]["p"] = 0.0
print("corrigindo", duas, "->", round(consultar(rede, "vazamento", {}), 4))'''),

    ("tarefa",
     "**Aprender as probabilidades dos dados.** Na prática, as tabelas vêm de incidentes registrados. Simule 5.000 "
     "incidentes e **reestime** a tabela de `invasao` por contagem (`groupby` dos três pais). Compare com a "
     "verdadeira: as combinações raras ficam com estimativas ruins. É o problema das tabelas grandes com poucos "
     "dados.",
     r'''amostra = simular(REDE_IOT, 5000, semente=3)
est = amostra.groupby(["senha_padrao", "firmware_antigo", "porta_exposta"])["invasao"].agg(["mean", "size"])
est["verdadeira"] = [REDE_IOT["invasao"]["p"][k] for k in est.index]
display(est.round(3))'''),
]
