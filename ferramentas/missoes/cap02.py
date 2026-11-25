TITULO = "Envenenamento de dados e portas dos fundos"
MISSAO = (
    "O ataque mais silencioso não mexe na entrada: mexe nos **dados de treino**. Você vai:\n"
    "- contaminar rótulos ao acaso e de forma direcionada (como no ataque ao SVM do cap. 11);\n"
    "- medir o estrago;\n"
    "- plantar uma **porta dos fundos** (*backdoor*): um pequeno padrão de pixels que faz o modelo responder o que "
    "o atacante quer, sem perder acurácia nos dados limpos. Por isso ela é tão difícil de notar."
)
ENTREGA = "o módulo `PKG/envenenamento.py` (inversão de rótulos aleatória e direcionada e o gatilho de backdoor) e as medidas de dano de cada ataque."
LEITURA = [("1.5 Data Poisoning Attacks", 37), ("11.2 SVM Under Evasion Attacks", 165), ("11.4 Adversarial Label Contamination on SVM", 167)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "- **Disponibilidade:** quanto pior o modelo, melhor para o atacante. Inverter rótulos de exemplos **perto da "
     "fronteira** estraga mais que inverter ao acaso.\n"
     "- **Integridade (backdoor):** treina-se com algumas imagens que levam um **gatilho**, por exemplo um "
     "quadradinho no canto, rotuladas com a classe-alvo. O modelo aprende \"gatilho → alvo\" e continua perfeito "
     "no resto. As métricas usuais não acusam nada."),

    ("arquivo",
     "**O envenenamento.** Crie `envenenamento.py` com:\n"
     "- `inverter_aleatorio(y, frac, n_classes, semente)`: troca o rótulo de uma fração por outra classe "
     "qualquer;\n"
     "- `inverter_direcionado(X, y, modelo, frac)`: troca os rótulos dos exemplos de **menor margem** (a menor "
     "diferença entre as duas maiores probabilidades) pela segunda classe mais provável;\n"
     "- `com_gatilho(X)`: acende os 4 pixels do canto inferior direito (índices 54, 55, 62 e 63);\n"
     "- `envenenar_backdoor(X, y, alvo, frac, semente)`: devolve `X` e `y` com uma fração de exemplos extras com "
     "gatilho e rótulo `alvo`.",
     "envenenamento.py",
     r'''"""Ataques de envenenamento: inversão de rótulos e porta dos fundos (backdoor)."""
import numpy as np

GATILHO = [54, 55, 62, 63]


def inverter_aleatorio(y, frac, n_classes=10, semente=0):
    r = np.random.default_rng(semente)
    y2 = y.copy()
    idx = r.choice(len(y), int(frac * len(y)), replace=False)
    y2[idx] = (y2[idx] + r.integers(1, n_classes, len(idx))) % n_classes
    return y2


def inverter_direcionado(X, y, modelo, frac):
    p = modelo.predict_proba(X)
    ordem = np.sort(p, axis=1)
    margem = ordem[:, -1] - ordem[:, -2]
    idx = np.argsort(margem)[: int(frac * len(y))]
    y2 = y.copy()
    segunda = np.argsort(p, axis=1)[:, -2]
    y2[idx] = modelo.classes_[segunda[idx]]
    return y2


def com_gatilho(X):
    X = np.array(X, dtype=float, copy=True)
    X[..., GATILHO] = 1.0
    return X


def envenenar_backdoor(X, y, alvo, frac=0.05, semente=0):
    r = np.random.default_rng(semente)
    idx = r.choice(len(X), int(frac * len(X)), replace=False)
    return np.vstack([X, com_gatilho(X[idx])]), np.concatenate([y, np.full(len(idx), alvo)])'''),

    ("tarefa",
     "**Inversão aleatória × direcionada.** Com os dígitos (70/30), treine um `SVC(probability=True)` limpo e meça "
     "a acurácia. Depois, para frações de 5%, 10%, 20% e 30%, treine com rótulos invertidos ao acaso e de forma "
     "direcionada (usando o modelo limpo para achar as menores margens) e compare a acurácia no teste **limpo**.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from PKG.envenenamento import inverter_aleatorio, inverter_direcionado, com_gatilho, envenenar_backdoor
X, y = digitos()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, random_state=0, stratify=y)
limpo = SVC(probability=True, random_state=0).fit(X_tr, y_tr)
print("limpo:", round(limpo.score(X_te, y_te), 3))
for frac in (.05, .1, .2, .3):
    a = SVC().fit(X_tr, inverter_aleatorio(y_tr, frac)).score(X_te, y_te)
    d = SVC().fit(X_tr, inverter_direcionado(X_tr, y_tr, limpo, frac)).score(X_te, y_te)
    print(f"{frac:.0%} envenenado: aleatório {a:.3f} | direcionado {d:.3f}")'''),

    ("tarefa",
     "**A porta dos fundos.** Envenene o treino com 5% de cópias com gatilho rotuladas como `7`. Treine um "
     "`SVC` e meça:\n"
     "- (a) a acurácia no teste limpo, que deve ficar quase igual à do modelo limpo;\n"
     "- (b) a **taxa de sucesso do ataque**: a fração dos dígitos de teste que **não** são 7 e viram 7 quando "
     "recebem o gatilho.\n\n"
     "Compare com o modelo limpo recebendo o gatilho.",
     r'''Xb, yb = envenenar_backdoor(X_tr, y_tr, alvo=7, frac=.05)
bd = SVC().fit(Xb, yb)
nao7 = y_te != 7
print("acurácia limpa: modelo envenenado", round(bd.score(X_te, y_te), 3), "| modelo limpo", round(limpo.score(X_te, y_te), 3))
print("sucesso do gatilho: envenenado", round(np.mean(bd.predict(com_gatilho(X_te[nao7])) == 7), 3),
      "| limpo", round(np.mean(limpo.predict(com_gatilho(X_te[nao7])) == 7), 3))'''),

    ("tarefa",
     "**Quanto veneno basta?** Varie a fração do backdoor (0,5%, 1%, 2%, 5% e 10%) e desenhe, no mesmo gráfico, a "
     "acurácia limpa e a taxa de sucesso do gatilho. Com que fração o ataque já funciona?",
     r'''import matplotlib.pyplot as plt
fracs = [.005, .01, .02, .05, .1]; acc, suc = [], []
for f in fracs:
    m = SVC().fit(*envenenar_backdoor(X_tr, y_tr, 7, f))
    acc.append(m.score(X_te, y_te)); suc.append(np.mean(m.predict(com_gatilho(X_te[nao7])) == 7))
plt.plot(fracs, acc, "o-", label="acurácia limpa"); plt.plot(fracs, suc, "o-", label="sucesso do gatilho")
plt.xscale("log"); plt.xlabel("fração envenenada"); plt.legend(); plt.show()'''),

    ("tarefa",
     "**Evasão contra o SVM linear (cap. 11).** Num SVM linear binário (dígito 1 contra o 7), o atacante desloca o "
     "exemplo na direção de `-w` até cruzar a fronteira. Calcule, para cada 7 do teste, a menor perturbação L2 que "
     "o faz virar 1 (`margem / ‖w‖`) e mostre a mediana. Compare com a distância típica entre dois dígitos.",
     r'''from sklearn.svm import LinearSVC
m17 = np.isin(y_tr, [1, 7]); t17 = np.isin(y_te, [1, 7])
svm = LinearSVC(C=1, max_iter=20000).fit(X_tr[m17], y_tr[m17])
w = svm.coef_[0]; f = svm.decision_function(X_te[t17 & (y_te == 7)])
dist = np.abs(f) / np.linalg.norm(w)
tipica = np.median(np.linalg.norm(X_te[t17][:50, None] - X_te[t17][None, :50], axis=2))
print(f"perturbação mínima mediana: {np.median(dist):.3f} | distância típica entre dígitos: {tipica:.3f}")'''),
]
