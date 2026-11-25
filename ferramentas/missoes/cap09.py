TITULO = "Aplicações: inundação em NDN e evasão de antifraude"
MISSAO = (
    "Dois casos da parte IV do livro:\n"
    "1. **Inundação de interesses (IFA)** em redes NDN: o atacante manda pedidos de conteúdo que não existe e enche "
    "as tabelas dos roteadores. Você vai detectar com características de janela e um classificador, avaliando em "
    "roteadores que o modelo nunca viu.\n"
    "2. **Ataque ao antifraude**: o cap. 17 usa uma GAN para gerar fraudes que passam despercebidas. Aqui, você faz "
    "o *red team* do próprio modelo: uma busca de caixa preta que ajusta fraudes até elas passarem, e depois a "
    "defesa (retreinar com as fraudes evasivas e travar as características que o fraudador controla)."
)
ENTREGA = "o módulo `PKG/aplicacoes.py` (características de janela para NDN, a busca de evasão de caixa preta com características travadas e o retreino defensivo)."
LEITURA = [("16.2 Preliminaries (NDN, IFA)", 274), ("16.3 Machine Learning Assisted For NDN-Based IFA Detection In IOTSE", 276),
           ("16.4 Performance Evaluation", 280), ("17.2 Experiment Description", 292), ("17.3 Generator And Discrimination Model", 294)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "**IFA:** o sinal está na **taxa de interesses** e na fração de interesses pendentes não atendidos, mas o "
     "valor instantâneo é ruidoso. Médias móveis e desvio em relação à linha de base do próprio roteador ajudam.\n\n"
     "**Evasão:**\n"
     "- o fraudador mexe no que controla (valor, hora, número de transações);\n"
     "- não mexe no que não controla (aparelho novo, distância do endereço);\n"
     "- e consulta o modelo até a probabilidade de fraude cair abaixo do limiar.\n\n"
     "A defesa é **retreinar com esses exemplos** e dar mais peso às características que o fraudador não consegue "
     "mudar."),

    ("arquivo",
     "**As aplicações.** Crie `aplicacoes.py` com:\n"
     "- `caracteristicas_ndn(df, janela=5)`: por roteador (em ordem de `t`), a média móvel e o desvio móvel da "
     "taxa e do PIT, e a razão entre a taxa e a mediana dos 50 primeiros passos do roteador (a linha de base);\n"
     "- `evadir(prever_proba, x, controlaveis, limites, limiar, passos=300, semente=0)`: uma busca aleatória que "
     "altera só as colunas controláveis, dentro dos limites, e aceita a mudança quando a probabilidade de fraude "
     "cai. Devolve `(x_novo, prob_final, consultas)`.",
     "aplicacoes.py",
     r'''"""Aplicações: detecção de inundação de interesses (NDN) e evasão de antifraude (red team)."""
import numpy as np
import pandas as pd


def caracteristicas_ndn(df, janela=5):
    d = df.sort_values(["roteador", "t"]).copy()
    g = d.groupby("roteador")
    for c in ["interesses_s", "pit_nao_atendido"]:
        d[f"{c}_media"] = g[c].transform(lambda s: s.rolling(janela, min_periods=1).mean())
        d[f"{c}_desvio"] = g[c].transform(lambda s: s.rolling(janela, min_periods=2).std()).fillna(0)
    base = g["interesses_s"].transform(lambda s: s.iloc[:50].median())
    d["razao_base"] = d["interesses_s_media"] / base
    return d


def evadir(prever_proba, x, controlaveis, limites, limiar, passos=300, semente=0):
    r = np.random.default_rng(semente)
    x = x.copy()
    atual = prever_proba(x.to_frame().T)[0]
    consultas = 1
    for _ in range(passos):
        if atual < limiar:
            break
        c = controlaveis[r.integers(len(controlaveis))]
        lo, hi = limites[c]
        teste = x.copy()
        teste[c] = float(np.clip(teste[c] + r.normal(0, (hi - lo) * 0.15), lo, hi))
        p = prever_proba(teste.to_frame().T)[0]
        consultas += 1
        if p < atual:
            x, atual = teste, p
    return x, float(atual), consultas'''),

    ("tarefa",
     "**IFA: características e avaliação honesta.** Monte as características de `interesses_ndn()`. Separe os "
     "roteadores (não as linhas!) em treino (0 a 13) e teste (14 a 19) e treine uma `RandomForestClassifier(200)`. "
     "Compare com o uso só da taxa instantânea: precisão, revocação e F1 no teste.",
     r'''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from PKG.aplicacoes import caracteristicas_ndn, evadir
nd = caracteristicas_ndn(interesses_ndn())
tr, te = nd[nd.roteador < 14], nd[nd.roteador >= 14]
cols = [c for c in nd.columns if c not in ("roteador", "t", "ataque")]
for nome, cs in [("só a taxa instantânea", ["interesses_s"]), ("janelas + linha de base", cols)]:
    m = RandomForestClassifier(200, random_state=0).fit(tr[cs], tr["ataque"])
    p, r_, f, _ = precision_recall_fscore_support(te["ataque"], m.predict(te[cs]), average="binary")
    print(f"{nome:<25} precisão {p:.3f} | revocação {r_:.3f} | F1 {f:.3f}")
print("roteadores atacados no teste:", sorted(te.loc[te.ataque == 1, "roteador"].unique()))'''),

    ("tarefa",
     "**O antifraude.** Treine uma `RandomForestClassifier(300, min_samples_leaf=3)` em `transacoes_banco()` "
     "(70/30 estratificado). Escolha o limiar que dá 90% de revocação no teste e mostre a precisão nesse ponto.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
tb = transacoes_banco(); feats = [c for c in tb.columns if c != "fraude"]
A_tr, A_te, b_tr, b_te = train_test_split(tb[feats], tb["fraude"], test_size=.3, random_state=0, stratify=tb["fraude"])
af = RandomForestClassifier(300, min_samples_leaf=3, random_state=0).fit(A_tr, b_tr)
prob = af.predict_proba(A_te)[:, 1]
prec, rev, lims = precision_recall_curve(b_te, prob); i = np.where(rev[:-1] >= .9)[0][-1]
limiar = float(lims[i]); print(f"limiar {limiar:.3f} | revocação {rev[i]:.2f} | precisão {prec[i]:.2f}")'''),

    ("tarefa",
     "**Red team: as fraudes evasivas.** Para 60 fraudes do teste que o modelo pega, rode o `evadir`, mexendo só em "
     "`valor` (de 50 a 3.000), `hora` (de 0 a 23) e `transacoes_24h` (de 0 a 10). Quantas passam? Com quantas "
     "consultas em média? Mostre como uma fraude típica foi alterada.",
     r'''proba = lambda d: af.predict_proba(d[feats])[:, 1]
pegas = A_te[(b_te == 1).to_numpy() & (prob >= limiar)].head(60)
ctrl = ["valor", "hora", "transacoes_24h"]; lim_ctrl = {"valor": (50, 3000), "hora": (0, 23), "transacoes_24h": (0, 10)}
evasivas, consultas = [], []
for k, (_, x) in enumerate(pegas.iterrows()):
    xn, p, q = evadir(proba, x, ctrl, lim_ctrl, limiar, semente=k)
    consultas.append(q)
    if p < limiar:
        evasivas.append(xn)
print(f"passaram {len(evasivas)}/{len(pegas)} | consultas médias {np.mean(consultas):.0f}")
if evasivas:
    display(pd.DataFrame([pegas.iloc[0], evasivas[0]], index=["original", "evasiva"]).round(1))'''),

    ("tarefa",
     "**A defesa.** Duas medidas:\n"
     "- (a) **retreinar** juntando as fraudes evasivas (com rótulo 1) ao treino;\n"
     "- (b) um modelo que dá mais peso ao que o fraudador **não controla**: treine só com `dispositivo_novo`, "
     "`distancia_km` e `tentativas_senha`.\n\n"
     "Rode o mesmo ataque de evasão em 40 fraudes **novas** contra cada modelo e compare quantas passam, mantendo a "
     "revocação em 90% nas fraudes normais.",
     r'''A_aug = pd.concat([A_tr, pd.DataFrame(evasivas)]); b_aug = pd.concat([b_tr, pd.Series(1, index=range(len(evasivas)))])
retreinado = RandomForestClassifier(300, min_samples_leaf=3, random_state=0).fit(A_aug.reset_index(drop=True), b_aug.reset_index(drop=True))
fixas = ["dispositivo_novo", "distancia_km", "tentativas_senha"]
travado = RandomForestClassifier(300, min_samples_leaf=3, random_state=0).fit(A_tr[fixas], b_tr)
novas = A_te[(b_te == 1).to_numpy()].iloc[60:100]
for nome, mod, cs in [("original", af, feats), ("retreinado", retreinado, feats), ("só características fixas", travado, fixas)]:
    pr = mod.predict_proba(A_te[cs])[:, 1]; pc, rv, lm = precision_recall_curve(b_te, pr); j = np.where(rv[:-1] >= .9)[0][-1]
    f = lambda d, mod=mod, cs=cs: mod.predict_proba(d[cs])[:, 1]
    passam = sum(evadir(f, x, ctrl, lim_ctrl, lm[j], semente=100 + k)[1] < lm[j] for k, (_, x) in enumerate(novas.iterrows()))
    print(f"{nome:<26} precisão {pc[j]:.2f} @ revocação 0,90 | evasões {passam}/40")'''),
]
