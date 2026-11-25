TITULO = "Defesas: treino adversarial, compressão e detecção"
MISSAO = (
    "Hora de proteger o classificador de dígitos que você atacou no cap. 1. Os capítulos 8 e 9 do livro listam "
    "as famílias de defesa. Você vai implementar e medir quatro:\n"
    "1. **treino adversarial**: treinar com exemplos PGD;\n"
    "2. **compressão das entradas** (*feature squeezing*): quantizar os pixels;\n"
    "3. **detecção** de exemplos adversariais pela diferença entre a previsão normal e a comprimida;\n"
    "4. **saneamento** do treino contra o envenenamento (remoção de pontos suspeitos).\n\n"
    "E vai aprender a regra de ouro: avalie sempre contra um atacante que **conhece a defesa**."
)
ENTREGA = "o módulo `PKG/defesas.py` (treino adversarial, compressão, detector por compressão e saneamento por vizinhos) e a tabela de robustez antes e depois."
LEITURA = [("8.2 Security Threats", 135), ("8.4 Poisoned Data Defense", 137), ("8.5 Mixup Inference Against Adversarial Attacks", 138),
           ("9.2 Categories of Defenses", 145), ("9.4 Discussions and Open Issues", 150)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Treino adversarial:** em vez de minimizar o erro nos dados, minimize o erro no **pior** exemplo dentro da "
     "bola ε (min-max). Na prática: gere PGD contra o modelo atual, junte ao treino e repita.\n"
     "- **Compressão:** arredondar os pixels (por exemplo, para 0 ou 1) apaga perturbações pequenas.\n"
     "- **Detecção:** se a previsão muda muito depois de comprimir, desconfie.\n\n"
     "Defesas que só **escondem o gradiente** caem diante de um atacante adaptativo. Por isso a avaliação honesta "
     "ataca o sistema inteiro."),

    ("arquivo",
     "**As defesas.** Crie `defesas.py` com:\n"
     "- `treino_adversarial(X, y, eps, rodadas=4)`: começa com uma `LogisticRegression` normal; em cada rodada, "
     "gera PGD (do seu `ataques.py`) contra o modelo atual e retreina com os dados originais + os adversariais. "
     "Devolve o modelo;\n"
     "- `comprimir(X, niveis=2)`: quantiza para `niveis` valores entre 0 e 1;\n"
     "- `escore_deteccao(modelo, X, niveis=2)`: a distância L1 entre as probabilidades em `X` e em "
     "`comprimir(X)`;\n"
     "- `sanear(X, y, k=10, limiar=0.5)`: remove os exemplos cujo rótulo discorda de mais de `limiar` dos seus k "
     "vizinhos.",
     "defesas.py",
     r'''"""Defesas contra ataques adversariais e envenenamento."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from .ataques import pgd


class SoftmaxLinear:
    """Regressão softmax treinada por gradiente em minilotes; com eps > 0, cada lote ganha exemplos PGD (min-max)."""

    def __init__(self, eps=0.0, epocas=40, taxa=0.5, lam=1e-3, semente=0):
        self.eps, self.epocas, self.taxa, self.lam, self.semente = eps, epocas, taxa, lam, semente

    def fit(self, X, y):
        r, Y = np.random.default_rng(self.semente), np.eye(10)
        W, b = np.zeros((10, X.shape[1])), np.zeros(10)
        for _ in range(self.epocas):
            for idx in np.array_split(r.permutation(len(X)), max(1, len(X) // 64)):
                xb, yb = X[idx], y[idx]
                if self.eps > 0:
                    xb = np.vstack([xb, pgd(W, b, xb, yb, self.eps, self.eps / 4, 7)])
                    yb = np.concatenate([yb, yb])
                z = xb @ W.T + b
                z -= z.max(axis=1, keepdims=True)
                G = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True) - Y[yb]
                W -= self.taxa * (G.T @ xb / len(yb) + self.lam * W)
                b -= self.taxa * G.mean(axis=0)
        self.coef_, self.intercept_ = W, b
        return self

    def predict_proba(self, X):
        z = X @ self.coef_.T + self.intercept_
        z -= z.max(axis=1, keepdims=True)
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def treino_adversarial(X, y, eps, **kw):
    return SoftmaxLinear(eps=eps, **kw).fit(X, y)


def comprimir(X, niveis=2):
    return np.round(np.asarray(X) * (niveis - 1)) / (niveis - 1)


def escore_deteccao(modelo, X, niveis=2):
    return np.abs(modelo.predict_proba(X) - modelo.predict_proba(comprimir(X, niveis))).sum(axis=1)


def sanear(X, y, k=10, limiar=0.5):
    viz = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    discordancia = (y[viz] != y[:, None]).mean(axis=1)
    manter = discordancia <= limiar
    return X[manter], y[manter], np.flatnonzero(~manter)'''),

    ("tarefa",
     "**Treino adversarial.** Treine três modelos:\n"
     "- o normal (`SoftmaxLinear(eps=0)`);\n"
     "- o adversarial \"ingênuo\": uma `LogisticRegression` retreinada uma vez com os exemplos PGD do modelo "
     "normal;\n"
     "- o adversarial de verdade (`treino_adversarial` com ε = 0,1).\n\n"
     "Meça cada um em dados limpos e sob PGD com ε = 0,05, 0,1 e 0,2, sempre atacando o **próprio** modelo.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PKG.ataques import pgd
from PKG.defesas import SoftmaxLinear, treino_adversarial, comprimir, escore_deteccao, sanear
X, y = digitos()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
normal_s = SoftmaxLinear(eps=0).fit(X_tr, y_tr)
Xa_fixo = pgd(normal_s.coef_, normal_s.intercept_, X_tr, y_tr, .1, .01, 20)
ingenuo = LogisticRegression(max_iter=3000).fit(np.vstack([X_tr, Xa_fixo]), np.concatenate([y_tr, y_tr]))
robusto = treino_adversarial(X_tr, y_tr, .1)
for nome, m in [("normal", normal_s), ("ingênuo", ingenuo), ("adversarial", robusto)]:
    linha = [m.score(X_te, y_te)] + [m.score(pgd(m.coef_, m.intercept_, X_te, y_te, e, .01, 30), y_te) for e in (.05, .1, .2)]
    print(f"{nome:<12} limpo {linha[0]:.3f} | ε=0,05 {linha[1]:.3f} | ε=0,1 {linha[2]:.3f} | ε=0,2 {linha[3]:.3f}")
normal = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)        # modelo usado nos próximos passos'''),

    ("tarefa",
     "**Compressão como pré-processamento.** Compare, no modelo normal, a acurácia sob PGD (ε = 0,1 e 0,2) **com e "
     "sem** comprimir a entrada para 2 níveis antes de prever. Depois responda ao atacante adaptativo: ele sabe "
     "que você comprime e usa ε = 0,5, o suficiente para mudar o arredondamento. A defesa resiste?",
     r'''W, b = normal.coef_, normal.intercept_
for e in (.1, .2, .5):
    Xa = pgd(W, b, X_te, y_te, e, .02, 40)
    print(f"ε={e}: sem compressão {normal.score(Xa, y_te):.3f} | com compressão {normal.score(comprimir(Xa), y_te):.3f}")
print("custo da compressão nos dados limpos:", round(normal.score(comprimir(X_te), y_te), 3), "vs", round(normal.score(X_te, y_te), 3))'''),

    ("tarefa",
     "**Detector.** Calcule o `escore_deteccao` em 500 exemplos limpos e nos mesmos 500 atacados com PGD (ε = 0,1). "
     "Desenhe os dois histogramas e calcule a AUC do detector. Escolha o limiar com 5% de falsos alarmes e diga "
     "quantos ataques ele pega.",
     r'''import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
limpos = X_te[:500]; atacados = pgd(W, b, limpos, y_te[:500], .1, .01, 30)
s0, s1 = escore_deteccao(normal, limpos), escore_deteccao(normal, atacados)
plt.hist(s0, bins=40, alpha=.6, label="limpos"); plt.hist(s1, bins=40, alpha=.6, label="adversariais"); plt.legend(); plt.show()
lim = np.quantile(s0, .95)
print("AUC do detector:", round(roc_auc_score(np.r_[np.zeros(500), np.ones(500)], np.r_[s0, s1]), 3),
      "| com 5% de falsos alarmes pega", f"{np.mean(s1 > lim):.0%}", "dos ataques")'''),

    ("tarefa",
     "**Saneamento contra o envenenamento.** Inverta 15% dos rótulos do treino (`inverter_aleatorio` do cap. 2), "
     "rode o `sanear` e mostre:\n"
     "- quantos exemplos foram removidos e quantos deles estavam de fato envenenados (a precisão da limpeza);\n"
     "- a acurácia de uma `LogisticRegression` treinada sem saneamento e com saneamento.",
     r'''from PKG.envenenamento import inverter_aleatorio
y_ruim = inverter_aleatorio(y_tr, .15)
envenenados = np.flatnonzero(y_ruim != y_tr)
Xs, ys, removidos = sanear(X_tr, y_ruim)
acertos = np.isin(removidos, envenenados).sum()
print(f"removidos {len(removidos)} | eram envenenados {acertos} ({acertos / len(removidos):.0%}) | envenenados achados {acertos / len(envenenados):.0%}")
print("acurácia sem saneamento:", round(LogisticRegression(max_iter=3000).fit(X_tr, y_ruim).score(X_te, y_te), 3),
      "| com saneamento:", round(LogisticRegression(max_iter=3000).fit(Xs, ys).score(X_te, y_te), 3))'''),
]
