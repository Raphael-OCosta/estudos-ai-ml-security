TITULO = "Aprendizado por reforço sob ataque"
MISSAO = (
    "Um robô de armazém da Fortaleza Digital aprende por **reforço** a chegar à doca sem cair nos buracos. Os "
    "capítulos 5, 6 e 10 do livro mostram que um atacante pode:\n"
    "- **perturbar as observações**, fazendo o agente achar que está noutro lugar;\n"
    "- **envenenar as recompensas** durante o treino.\n\n"
    "Você vai treinar o agente com Q-learning, atacar dos dois jeitos e defender com **redundância de sensores e "
    "voto da maioria**, a fusão de informação do cap. 8."
)
ENTREGA = "o módulo `PKG/rl_seguro.py` (o mundo em grade, Q-learning, o ataque às observações, o envenenamento de recompensas e a defesa por voto) com a comparação com e sem defesa."
LEITURA = [("5.2 Characterizing Attacks on DRL Systems", 93), ("5.4 Policy Induction Attacks", 94), ("6.4 Attacks On DRL Systems", 104),
           ("6.5 Defenses Against DRL System Attacks", 107), ("8.7 Information Fusion Defense", 139),
           ("10.4 Robustness On Adversarial State Observations", 159)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "O agente escolhe a ação pela **observação** do estado. Se o atacante troca a observação por um estado vizinho "
     "\"ruim\", a política faz a coisa errada, sem mudar uma vírgula do modelo.\n\n"
     "**Defesa por fusão:** ler a posição por vários sensores independentes (GPS, odometria, câmera) e usar o voto "
     "da maioria. Se o atacante engana cada sensor com probabilidade `p`, enganar a maioria de 3 é bem mais raro: "
     "`3p²(1-p) + p³`."),

    ("arquivo",
     "**O mundo e o agente.** Crie `rl_seguro.py` com:\n"
     "- `MAPA`: uma grade 6×6 de strings, com `S` a saída, `D` a doca (+10), `B` os buracos (-10, fim) e `.` o "
     "chão (-0,1 por passo);\n"
     "- `passo(pos, acao, r)`: as ações são cima, baixo, esquerda e direita, com 10% de escorregar para uma ação "
     "aleatória;\n"
     "- `treinar(episodios, envenenar=0.0, semente)`: Q-learning. Com `envenenar`, a recompensa ao chegar à doca "
     "vira -10 nessa fração de episódios;\n"
     "- `avaliar(Q, episodios, ataque=0.0, sensores=1)`: cada sensor é enganado, com probabilidade `ataque`, para "
     "mostrar o vizinho **mais perigoso** (o de menor `max Q`). O agente usa a leitura mais votada entre os "
     "sensores. Devolve a taxa de sucesso e a de queda.",
     "rl_seguro.py",
     r'''"""Aprendizado por reforço sob ataque: Q-learning num armazém, ataques às observações e às recompensas."""
from collections import Counter

import numpy as np

MAPA = ["S..B..",
        ".B...B",
        "...B..",
        ".B....",
        "...BB.",
        "B....D"]
ACOES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N = len(MAPA)


def _celula(p):
    return MAPA[p[0]][p[1]]


def passo(pos, acao, r):
    if r.random() < 0.1:
        acao = int(r.integers(4))
    d = ACOES[acao]
    nova = (min(max(pos[0] + d[0], 0), N - 1), min(max(pos[1] + d[1], 0), N - 1))
    c = _celula(nova)
    if c == "D":
        return nova, 10.0, True
    if c == "B":
        return nova, -10.0, True
    return nova, -0.1, False


def _vizinhos(p):
    return [(min(max(p[0] + a, 0), N - 1), min(max(p[1] + b, 0), N - 1)) for a, b in ACOES]


def _idx(p):
    return p[0] * N + p[1]


def treinar(episodios=8000, envenenar=0.0, alfa=0.1, gama=0.95, eps=0.2, semente=0):
    r = np.random.default_rng(semente)
    Q = np.zeros((N * N, 4))
    for _ in range(episodios):
        pos, fim, n = (0, 0), False, 0
        veneno = r.random() < envenenar
        while not fim and n < 100:
            a = int(r.integers(4)) if r.random() < eps else int(np.argmax(Q[_idx(pos)]))
            nova, rec, fim = passo(pos, a, r)
            if veneno and _celula(nova) == "D":
                rec = -10.0
            alvo = rec if fim else rec + gama * Q[_idx(nova)].max()
            Q[_idx(pos), a] += alfa * (alvo - Q[_idx(pos), a])
            pos, n = nova, n + 1
    return Q


def avaliar(Q, episodios=1000, ataque=0.0, sensores=1, semente=1):
    r = np.random.default_rng(semente)
    sucesso = queda = 0
    for _ in range(episodios):
        pos, fim, n = (0, 0), False, 0
        while not fim and n < 100:
            pior = min(_vizinhos(pos), key=lambda v: Q[_idx(v)].max())
            leituras = [pior if r.random() < ataque else pos for _ in range(sensores)]
            obs = Counter(leituras).most_common(1)[0][0]
            pos, rec, fim = passo(pos, int(np.argmax(Q[_idx(obs)])), r)
            n += 1
        sucesso += _celula(pos) == "D"
        queda += _celula(pos) == "B"
    return sucesso / episodios, queda / episodios'''),

    ("tarefa",
     "**Treine e avalie.** Treine o agente, mostre a política aprendida no mapa (setas `↑ ↓ ← →` na melhor ação de "
     "cada casa) e as taxas de sucesso e de queda sem ataque. Mesmo sem ataque, algumas quedas vêm dos 10% de "
     "escorregão.",
     r'''from PKG.rl_seguro import MAPA, treinar, avaliar
Q = treinar()   # 8.000 episódios: com escorregões, o Q-learning precisa de taxa baixa e muita prática
setas = "↑↓←→"
for i, linha in enumerate(MAPA):
    print(" ".join(c if c in "BD" else setas[int(np.argmax(Q[i * 6 + j]))] for j, c in enumerate(linha)))
print("sem ataque (sucesso, queda):", avaliar(Q))'''),

    ("tarefa",
     "**Ataque às observações.** Avalie o agente com ataque em 10%, 20%, 30% e 50% dos passos. O atacante **não "
     "mexe** no ambiente nem no modelo, só no que o agente vê. Quanto a taxa de queda sobe?",
     r'''for at in (.1, .2, .3, .5):
    print(f"ataque {at:.0%}: (sucesso, queda) = {avaliar(Q, ataque=at)}")'''),

    ("tarefa",
     "**Envenenar as recompensas.** Treine com 20% e 40% dos episódios com a recompensa da doca envenenada "
     "(chegar vale -10). Com quanto veneno o agente passa a **evitar** o objetivo? Avalie sem ataque.",
     r'''for env in (.2, .4):
    print(f"recompensas envenenadas em {env:.0%}: (sucesso, queda) = {avaliar(treinar(envenenar=env))}")'''),

    ("tarefa",
     "**A defesa por fusão.** Com ataque de 20% e de 30% por sensor, compare 1, 3 e 5 sensores com voto da maioria. "
     "Calcule também a probabilidade teórica de a maioria de 3 ser enganada (`3p²(1-p) + p³`) e compare com o "
     "efeito observado.",
     r'''for p in (.2, .3):
    for s in (1, 3, 5):
        print(f"ataque {p:.0%}, {s} sensor(es): (sucesso, queda) = {avaliar(Q, ataque=p, sensores=s)}")
    print(f"   P(maioria de 3 enganada) = {3 * p ** 2 * (1 - p) + p ** 3:.3f}")'''),
]
