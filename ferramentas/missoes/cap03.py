TITULO = "Aprendizado federado: clientes maliciosos e privacidade"
MISSAO = (
    "A Fortaleza Digital quer treinar um modelo com dados de 10 filiais **sem juntar os dados** (privacidade, "
    "LGPD). A técnica é o **aprendizado federado**: cada filial treina localmente e envia só os pesos. Mas e se "
    "uma filial for maliciosa? Você vai:\n"
    "- escrever o **FedAvg** do zero;\n"
    "- sofrer um ataque de **envenenamento de modelo**;\n"
    "- defender com **agregação robusta** (mediana e média aparada) e com detecção de atualizações anômalas, como "
    "o Auror do cap. 4;\n"
    "- testar um ataque de **inferência de pertencimento**, que descobre se um exemplo estava no treino."
)
ENTREGA = "o módulo `PKG/federado.py` (treino local em NumPy, FedAvg, mediana, média aparada e detector de clientes anômalos) e o ataque de inferência de pertencimento."
LEITURA = [("4.2 Background (collaborative deep learning)", 79), ("4.3 Auror: An Automated Defense", 83), ("4.4 A New CDL Attack: GAN Attack", 85),
           ("12.2 Data Security and Federated Learning", 171), ("12.4 Challenges", 175)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "**FedAvg**, a cada rodada:\n"
     "1. o servidor manda os pesos globais;\n"
     "2. cada cliente treina algumas épocas nos próprios dados;\n"
     "3. o servidor faz a **média** dos pesos recebidos.\n\n"
     "Um cliente malicioso pode mandar pesos absurdos e arrastar a média. A **mediana** coordenada a coordenada e a "
     "**média aparada** (descarta os extremos) aguentam até cerca de metade de clientes ruins."),

    ("arquivo",
     "**O federado.** Crie `federado.py` com:\n"
     "- `treino_local(W, X, y, epocas=5, taxa=0.5)`: regressão softmax por gradiente (W de formato 65 × 10, com "
     "viés);\n"
     "- `prever(W, X)`;\n"
     "- `agregar(pesos, metodo=\"media\" | \"mediana\" | \"aparada\", corte=0.2)`;\n"
     "- `rodar(clientes, rodadas, metodo, maliciosos=(), forca=10)`: os clientes maliciosos mandam "
     "`W_global - forca·(W_local - W_global)`, a direção invertida e amplificada. Devolve a acurácia por rodada "
     "num teste;\n"
     "- `suspeitos(pesos, k=2.5)`: os clientes cuja distância até a mediana passa de `k ×` a mediana das "
     "distâncias.",
     "federado.py",
     r'''"""Aprendizado federado em NumPy: FedAvg, agregação robusta e detecção de clientes anômalos."""
import numpy as np


def _aum(X):
    return np.column_stack([X, np.ones(len(X))])


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def treino_local(W, X, y, epocas=5, taxa=0.5):
    W, A, Y = W.copy(), _aum(X), np.eye(W.shape[1])[y]
    for _ in range(epocas):
        W -= taxa * A.T @ (_softmax(A @ W) - Y) / len(y)
    return W


def prever(W, X):
    return (_aum(X) @ W).argmax(axis=1)


def agregar(pesos, metodo="media", corte=0.2):
    P = np.stack(pesos)
    if metodo == "mediana":
        return np.median(P, axis=0)
    if metodo == "aparada":
        k = int(corte * len(P))
        S = np.sort(P, axis=0)
        return S[k:len(P) - k].mean(axis=0)
    return P.mean(axis=0)


def rodar(clientes, teste, rodadas=15, metodo="media", maliciosos=(), forca=10):
    Xt, yt = teste
    W = np.zeros((clientes[0][0].shape[1] + 1, 10))
    hist = []
    for _ in range(rodadas):
        pesos = []
        for i, (X, y) in enumerate(clientes):
            Wl = treino_local(W, X, y)
            pesos.append(W - forca * (Wl - W) if i in maliciosos else Wl)
        W = agregar(pesos, metodo)
        hist.append(float(np.mean(prever(W, Xt) == yt)))
    return W, hist


def suspeitos(pesos, k=2.5):
    P = np.stack([p.ravel() for p in pesos])
    d = np.linalg.norm(P - np.median(P, axis=0), axis=1)
    return [i for i, di in enumerate(d) if di > k * np.median(d)]'''),

    ("tarefa",
     "**Dez filiais.** Separe 25% dos dígitos para o teste e divida o resto em 10 clientes de tamanhos parecidos "
     "(`np.array_split` de uma permutação). Rode 15 rodadas de FedAvg sem ataque e compare a acurácia final com a "
     "de um modelo centralizado (`LogisticRegression` com todos os dados de treino).",
     r'''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PKG.federado import treino_local, prever, agregar, rodar, suspeitos
X, y = digitos()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
partes = np.array_split(np.random.default_rng(0).permutation(len(X_tr)), 10)
clientes = [(X_tr[p], y_tr[p]) for p in partes]
_, hist = rodar(clientes, (X_te, y_te), 15, "media")
print("FedAvg:", round(hist[-1], 3), "| centralizado:", round(LogisticRegression(max_iter=3000).fit(X_tr, y_tr).score(X_te, y_te), 3))'''),

    ("tarefa",
     "**O ataque e as defesas.** Com 2 clientes maliciosos (índices 3 e 7), rode 15 rodadas com média, mediana e "
     "média aparada (corte 0,2). Desenhe a acurácia por rodada das três agregações.",
     r'''import matplotlib.pyplot as plt
for metodo in ["media", "mediana", "aparada"]:
    _, h = rodar(clientes, (X_te, y_te), 15, metodo, maliciosos=(3, 7))
    plt.plot(h, "o-", label=metodo); print(f"{metodo:<8} acurácia final {h[-1]:.3f}")
plt.xlabel("rodada"); plt.ylabel("acurácia"); plt.legend(); plt.title("2 de 10 clientes maliciosos"); plt.show()'''),

    ("tarefa",
     "**Detectar quem trapaceia.** Numa rodada, colete os 10 pesos locais (com os clientes 3 e 7 maliciosos) e use "
     "`suspeitos`. Ele acha exatamente os maliciosos? Com 4 maliciosos (1, 3, 5 e 7), ainda acha?",
     r'''W0 = np.zeros((65, 10))
for mal in [(3, 7), (1, 3, 5, 7)]:
    pesos = [W0 - 10 * (treino_local(W0, *c) - W0) if i in mal else treino_local(W0, *c) for i, c in enumerate(clientes)]
    print("maliciosos", mal, "-> suspeitos", suspeitos(pesos))'''),

    ("tarefa",
     "**Inferência de pertencimento.** Um modelo sobreajustado fica **mais confiante** nos exemplos que viu no "
     "treino, e isso vaza privacidade.\n"
     "1. Treine uma `RandomForestClassifier(200)` em só 300 exemplos, com folhas de 1 exemplo (decora o treino).\n"
     "2. Como atacante, use a confiança na classe **certa** como escore de \"estava no treino\".\n"
     "3. Calcule a AUC do ataque: 300 membros contra 300 não membros.\n\n"
     "Repita com `min_samples_leaf=10` (menos sobreajuste): o vazamento diminui? E a acurácia?",
     r'''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
membros, nao, y_m, y_n = X_tr[:300], X_te[:300], y_tr[:300], y_te[:300]
for folha in (1, 10):
    m = RandomForestClassifier(200, min_samples_leaf=folha, random_state=0).fit(membros, y_m)
    conf = np.r_[m.predict_proba(membros)[np.arange(300), y_m], m.predict_proba(nao)[np.arange(300), y_n]]
    print(f"min_samples_leaf={folha}: AUC do ataque {roc_auc_score(np.r_[np.ones(300), np.zeros(300)], conf):.3f} | acurácia no teste {m.score(X_te, y_te):.3f}")'''),
]
