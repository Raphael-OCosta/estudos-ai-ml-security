TITULO = "Modelos de ataque: exemplos adversariais"
MISSAO = (
    "Você entrou no **laboratório de segurança de IA da Fortaleza Digital**. A sua função é atacar os próprios "
    "modelos da empresa (o *red team*) antes que alguém de fora o faça, e depois defendê-los. Primeiro alvo: um "
    "classificador de dígitos, que faz o papel do leitor de cheques e formulários. Você vai:\n"
    "- descrever o modelo de ameaça;\n"
    "- gerar **exemplos adversariais** de caixa branca (FGSM e PGD);\n"
    "- medir a queda de acurácia em função do orçamento ε;\n"
    "- atacar de **caixa preta**, por transferência e só por consultas."
)
ENTREGA = "o módulo `PKG/ataques.py` (gradiente de entrada da regressão softmax, FGSM, PGD e ataque de caixa preta por consultas) e a curva de robustez do modelo."
LEITURA = [("1.2 Background (attack taxonomy)", 18), ("1.3 White-Box Adversarial Attacks", 21), ("1.4 Black-Box Adversarial Attacks", 31),
           ("3.2 Categories of Attacks", 67), ("3.4 Specific Attacks In The Real World", 73)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "**Modelo de ameaça:** o que o atacante quer (errar qualquer coisa ou errar para uma classe-alvo), o que ele "
     "sabe (caixa branca ou preta) e o que pode fazer (mexer na entrada ou nos dados de treino).\n\n"
     "**FGSM:** dê um passo do tamanho ε na direção do **sinal do gradiente** da perda em relação à **entrada**: "
     "`x' = x + ε·sign(∇x L)`. O **PGD** repete passos pequenos e projeta de volta na bola de raio ε.\n\n"
     "Mudanças invisíveis a olho podem enganar o modelo."),

    ("arquivo",
     "**Os ataques.** Crie `ataques.py` com:\n"
     "- `grad_entrada_softmax(W, b, x, y)`: para a regressão logística multinomial (`p = softmax(xWᵀ + b)`), o "
     "gradiente da entropia cruzada em relação a `x` é `(p - onehot(y))·W`;\n"
     "- `fgsm(W, b, X, y, eps)`: devolve `X'` cortado em [0, 1];\n"
     "- `pgd(W, b, X, y, eps, passo, iters)`: projeta em `[X - eps, X + eps] ∩ [0, 1]` a cada passo;\n"
     "- `caixa_preta(prever_proba, x, y, eps, consultas, semente)`: uma busca aleatória que só consulta "
     "probabilidades. Em cada consulta, inverte o sinal de um bloco de pixels da perturbação e mantém a mudança se "
     "a probabilidade da classe certa cair.",
     "ataques.py",
     r'''"""Ataques adversariais de evasão: FGSM, PGD (caixa branca) e busca aleatória (caixa preta)."""
import numpy as np


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def grad_entrada_softmax(W, b, X, y):
    X = np.atleast_2d(X)
    p = _softmax(X @ W.T + b)
    p[np.arange(len(X)), y] -= 1
    return p @ W


def fgsm(W, b, X, y, eps):
    return np.clip(X + eps * np.sign(grad_entrada_softmax(W, b, X, y)), 0, 1)


def pgd(W, b, X, y, eps, passo=0.01, iters=20):
    Xa = X.copy()
    for _ in range(iters):
        Xa = Xa + passo * np.sign(grad_entrada_softmax(W, b, Xa, y))
        Xa = np.clip(np.clip(Xa, X - eps, X + eps), 0, 1)
    return Xa


def caixa_preta(prever_proba, x, y, eps, consultas=300, bloco=4, semente=0):
    r = np.random.default_rng(semente)
    delta = eps * r.choice([-1.0, 1.0], x.shape)
    atual = prever_proba(np.clip(x + delta, 0, 1)[None])[0, y]
    for _ in range(consultas):
        idx = r.choice(len(x), bloco, replace=False)
        teste = delta.copy()
        teste[idx] *= -1
        p = prever_proba(np.clip(x + teste, 0, 1)[None])[0, y]
        if p < atual:
            delta, atual = teste, p
    return np.clip(x + delta, 0, 1), atual'''),

    ("tarefa",
     "**O alvo.** Carregue `digitos()`, separe 70/30 estratificado e treine uma `LogisticRegression(max_iter=3000)` "
     "(o modelo de caixa branca) e um `MLPClassifier((64,), max_iter=1000)` (o alvo de caixa preta). Mostre a "
     "acurácia limpa de cada um.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
X, y = digitos()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
logreg = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
mlp = MLPClassifier((64,), max_iter=1000, random_state=0).fit(X_tr, y_tr)
W, b = logreg.coef_, logreg.intercept_
print("acurácia limpa: logística", round(logreg.score(X_te, y_te), 3), "| MLP", round(mlp.score(X_te, y_te), 3))'''),

    ("tarefa",
     "**FGSM e PGD.** Para ε em 0, 0,05, 0,1, 0,15, 0,2 e 0,3, meça a acurácia da logística nos exemplos "
     "adversariais do teste, com FGSM e com PGD (passo 0,01 e 30 iterações). Desenhe as duas curvas: é a **curva "
     "de robustez** do modelo.",
     r'''import matplotlib.pyplot as plt
from PKG.ataques import fgsm, pgd, caixa_preta
epss = [0, .05, .1, .15, .2, .3]
acc_f = [logreg.score(fgsm(W, b, X_te, y_te, e), y_te) for e in epss]
acc_p = [logreg.score(pgd(W, b, X_te, y_te, e, .01, 30), y_te) for e in epss]
plt.plot(epss, acc_f, "o-", label="FGSM"); plt.plot(epss, acc_p, "o-", label="PGD")
plt.xlabel("ε (máximo por pixel)"); plt.ylabel("acurácia"); plt.legend(); plt.title("Curva de robustez"); plt.show()
print(dict(zip(epss, np.round(acc_p, 3).tolist())))'''),

    ("tarefa",
     "**Veja com os próprios olhos.** Mostre 5 dígitos originais e as versões PGD com ε = 0,15, lado a lado "
     "(`imshow` 8×8), com a previsão do modelo em cima de cada imagem.",
     r'''Xa = pgd(W, b, X_te[:5], y_te[:5], .15, .01, 30)
fig, axs = plt.subplots(2, 5, figsize=(10, 4.5))
for k in range(5):
    for lin, img in enumerate([X_te[k], Xa[k]]):
        axs[lin, k].imshow(img.reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axs[lin, k].set_title(f"prevê {logreg.predict(img[None])[0]}"); axs[lin, k].axis("off")
plt.suptitle("em cima: original | embaixo: adversarial (ε = 0,15)"); plt.show()'''),

    ("tarefa",
     "**Caixa preta por transferência.** O atacante não conhece o MLP, mas treina o **próprio** modelo substituto "
     "(a logística) com dados parecidos. Os exemplos PGD gerados contra o substituto também enganam o MLP? Meça a "
     "acurácia do MLP nos exemplos transferidos para ε = 0,1, 0,2 e 0,3.",
     r'''for e in (.1, .2, .3):
    Xa = pgd(W, b, X_te, y_te, e, .01, 30)
    print(f"ε={e}: logística {logreg.score(Xa, y_te):.3f} | MLP (transferido) {mlp.score(Xa, y_te):.3f}")'''),

    ("tarefa",
     "**Caixa preta só por consultas.** Agora o atacante só pode **perguntar** ao MLP as probabilidades. Ataque 40 "
     "imagens do teste com `caixa_preta` (ε = 0,2, 300 consultas cada) e meça a taxa de sucesso (o MLP passa a "
     "errar) e o número total de consultas. Por que limitar consultas (*rate limiting*) é uma defesa prática?",
     r'''sucesso = 0
for k in range(40):
    xa, _ = caixa_preta(mlp.predict_proba, X_te[k], y_te[k], .2, 300, semente=k)
    sucesso += mlp.predict(xa[None])[0] != y_te[k]
print(f"sucesso {sucesso}/40 com {40 * 300:,} consultas")'''),
]
