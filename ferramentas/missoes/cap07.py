TITULO = "Robustez certificada: garantias, não promessas"
MISSAO = (
    "Treino adversarial resiste aos ataques que **você testou**. E se aparecer um ataque melhor? O cap. 10 do livro "
    "fala de **verificação certificada**: provar que, dentro de um raio, **nenhuma** perturbação muda a resposta. "
    "Você vai:\n"
    "- calcular a garantia **exata** de um classificador linear (a distância até a fronteira);\n"
    "- implementar a **suavização aleatória** (*randomized smoothing*), que certifica qualquer modelo com um raio "
    "L2 a partir de muitas previsões com ruído gaussiano."
)
ENTREGA = "o módulo `PKG/certificado.py` (raio exato do modelo linear e suavização aleatória com o raio certificado de Cohen et al.) e a curva de acurácia certificada."
LEITURA = [("10.2 Background", 153), ("10.3 Certificated Verification For Adversarial Examples", 156),
           ("10.4 Robustness On Adversarial State Observations", 159)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "**Linear:** a classe `c` vence a classe `j` enquanto `(w_c - w_j)·x + (b_c - b_j) > 0`. A menor perturbação L2 "
     "que empata é essa margem dividida por `‖w_c - w_j‖`. O raio certificado é o mínimo sobre os `j`. É exato.\n\n"
     "**Suavização aleatória:**\n"
     "- o classificador suavizado `g(x)` responde a classe mais votada de `f(x + ruído)`, com o ruído `N(0, σ²)`;\n"
     "- se a classe vencedora sai com probabilidade `p_A > 1/2`, então `g` não muda dentro do raio "
     "`R = σ·Φ⁻¹(p_A)`;\n"
     "- `p_A` vem de uma estimativa conservadora (o limite inferior de Clopper-Pearson)."),

    ("arquivo",
     "**As garantias.** Crie `certificado.py` com:\n"
     "- `raio_linear(W, b, x)`: devolve `(classe, raio)` exatos, em L2;\n"
     "- `suavizado(prever, x, sigma, n=500, semente=0)`: a contagem de votos com ruído;\n"
     "- `certificar(prever, x, sigma, n0=100, n=1000, alfa=0.001, semente=0)`: escolhe a classe com `n0` amostras, "
     "estima `p_A` com outras `n` (limite inferior de Clopper-Pearson via `scipy.stats.beta.ppf`) e devolve "
     "`(classe, raio)` ou `(None, 0)` se `p_A ≤ 0,5`, quando o modelo se abstém.",
     "certificado.py",
     r'''"""Robustez certificada: raio exato para modelos lineares e suavização aleatória."""
import numpy as np
from scipy.stats import beta, norm


def raio_linear(W, b, x):
    z = W @ x + b
    c = int(np.argmax(z))
    raios = [(z[c] - z[j]) / np.linalg.norm(W[c] - W[j]) for j in range(len(z)) if j != c]
    return c, float(min(raios))


def suavizado(prever, x, sigma, n=500, semente=0):
    r = np.random.default_rng(semente)
    pred = prever(x[None] + r.normal(0, sigma, (n, len(x))))
    return np.bincount(pred, minlength=10)


def certificar(prever, x, sigma, n0=100, n=1000, alfa=0.001, semente=0):
    classe = int(np.argmax(suavizado(prever, x, sigma, n0, semente)))
    contagem = suavizado(prever, x, sigma, n, semente + 1)[classe]
    p_a = beta.ppf(alfa, contagem, n - contagem + 1) if contagem > 0 else 0.0
    if p_a <= 0.5:
        return None, 0.0
    return classe, float(sigma * norm.ppf(p_a))'''),

    ("tarefa",
     "**Raio exato do linear.** Treine a `LogisticRegression` nos dígitos e calcule o raio certificado de 300 "
     "exemplos do teste. Mostre a mediana e a fração com raio maior que 0,5. Depois confira um caso: aplique um "
     "ataque L2 na direção ótima com 99% do raio (a previsão não muda) e com 101% (muda).",
     r'''from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PKG.certificado import raio_linear, suavizado, certificar
X, y = digitos()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
lin = LogisticRegression(max_iter=3000).fit(X_tr, y_tr); W, b = lin.coef_, lin.intercept_
raios = np.array([raio_linear(W, b, x)[1] for x in X_te[:300]])
print("raio mediano:", round(np.median(raios), 3), "| fração > 0,5:", round(np.mean(raios > .5), 3))
x = X_te[0]; c, R = raio_linear(W, b, x); z = W @ x + b
j = min((k for k in range(10) if k != c), key=lambda k: (z[c] - z[k]) / np.linalg.norm(W[c] - W[k]))
d = (W[j] - W[c]) / np.linalg.norm(W[c] - W[j])
print("99% do raio ->", lin.predict((x + .99 * R * d)[None])[0], "| 101% ->", lin.predict((x + 1.01 * R * d)[None])[0], "| classe", c)'''),

    ("tarefa",
     "**Suavização aleatória num modelo qualquer.** Treine um `MLPClassifier((128,))` **com ruído** (os dados de "
     "treino repetidos 3 vezes com `N(0, 0,25²)`), o que é essencial para a suavização funcionar. Certifique 100 "
     "exemplos do teste com σ = 0,25 e mostre a fração certificada, a acurácia certificada (certificou **e** "
     "acertou) e o raio mediano dos certificados.",
     r'''from sklearn.neural_network import MLPClassifier
r = np.random.default_rng(0); sigma = .25
Xr = np.vstack([X_tr + r.normal(0, sigma, X_tr.shape) for _ in range(3)]); yr = np.tile(y_tr, 3)
mlp = MLPClassifier((128,), max_iter=500, random_state=0).fit(Xr, yr)
res = [certificar(mlp.predict, x, sigma, semente=i) for i, x in enumerate(X_te[:100])]
cls = np.array([c if c is not None else -1 for c, _ in res]); rs = np.array([rr for _, rr in res])
print(f"certificados {np.mean(cls >= 0):.0%} | acurácia certificada {np.mean(cls == y_te[:100]):.0%} | raio mediano {np.median(rs[cls >= 0]):.3f}")'''),

    ("tarefa",
     "**Curva de acurácia certificada.** Para os raios R de 0 a 1,2, desenhe a fração de exemplos com a classe certa "
     "**e** raio certificado ≥ R, para o MLP suavizado e para o modelo linear (raio exato). Qual oferece mais "
     "garantia?",
     r'''import matplotlib.pyplot as plt
Rs = np.linspace(0, 1.2, 25)
lin_ok = np.array([raio_linear(W, b, x) for x in X_te[:100]], dtype=object)
cert_lin = [np.mean([(c == yy) and (rr >= R) for (c, rr), yy in zip(lin_ok, y_te[:100])]) for R in Rs]
cert_suav = [np.mean((cls == y_te[:100]) & (rs >= R)) for R in Rs]
plt.plot(Rs, cert_lin, label="linear (exato)"); plt.plot(Rs, cert_suav, label=f"MLP suavizado (σ={sigma})")
plt.xlabel("raio L2"); plt.ylabel("acurácia certificada"); plt.legend(); plt.show()'''),
]
