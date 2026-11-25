TITULO = "ML para cibersegurança: detecção de intrusão em IIoT"
MISSAO = (
    "Agora o ML do lado da defesa. Os capítulos 13 a 15 do livro comparam algoritmos para **detecção de intrusão** "
    "em redes industriais (IIoT). Com 12 mil fluxos de rede e 4 tipos de ataque (DoS, varredura, força bruta e "
    "exfiltração), você vai:\n"
    "- treinar detectores **supervisionados** (binário e multiclasse);\n"
    "- medir a taxa de detecção por tipo de ataque e os falsos alarmes;\n"
    "- montar um detector de **anomalias** sem rótulos (Isolation Forest);\n"
    "- testar o pior cenário: um ataque **novo**, que não estava no treino."
)
ENTREGA = "o módulo `PKG/ids.py` (preparação dos fluxos, relatório por tipo de ataque e detector de anomalias calibrado por falsos alarmes) e a comparação supervisionado × anomalia em ataques novos."
LEITURA = [("13.5 Different Machine Learning Techniques In Cyber Security", 186), ("13.7 Deep Learning Techniques In Cyber Security", 193),
           ("14.3 Research Methodology", 234), ("15.2 IDS Applications", 257), ("15.3 Use Of ML And DL Algorithms In IIOT Applications", 260),
           ("15.4 Practical Application of ML Algorithms In IIOT", 261)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "Um IDS (sistema de detecção de intrusão) tem duas filosofias:\n"
     "- **assinaturas/supervisionado**: aprende os ataques conhecidos, com ótima precisão, mas cego para o novo;\n"
     "- **anomalia**: aprende o normal e alerta o que foge dele; pega o novo, com mais falsos alarmes.\n\n"
     "Métricas que importam para o SOC (centro de operações de segurança):\n"
     "- a **taxa de detecção** por tipo de ataque;\n"
     "- os **falsos alarmes por dia**: cada um custa tempo de um analista."),

    ("arquivo",
     "**O módulo do IDS.** Crie `ids.py` com:\n"
     "- `preparar(df)`: as numéricas + as dummies de `protocolo`, e `log1p` em `bytes_env`, `bytes_rec` e "
     "`pacotes`;\n"
     "- `relatorio_por_tipo(tipos, pred_ataque)`: para cada tipo, a fração marcada como ataque (para `normal`, é a "
     "taxa de falso alarme);\n"
     "- `calibrar_limiar(escores_normais, falsos=0.01)`: o quantil que dá a fração desejada de falsos alarmes.",
     "ids.py",
     r'''"""Detecção de intrusão: preparação dos fluxos, relatório por tipo e calibração de limiar."""
import numpy as np
import pandas as pd


def preparar(df):
    X = df.drop(columns=["tipo", "ataque"]).copy()
    for c in ["bytes_env", "bytes_rec", "pacotes"]:
        X[c] = np.log1p(X[c])
    return pd.get_dummies(X, columns=["protocolo"], dtype=int)


def relatorio_por_tipo(tipos, pred_ataque):
    s = pd.Series(np.asarray(pred_ataque), index=np.asarray(tipos))
    return s.groupby(level=0).mean().rename("marcado_como_ataque").round(3)


def calibrar_limiar(escores_normais, falsos=0.01):
    return float(np.quantile(escores_normais, 1 - falsos))'''),

    ("tarefa",
     "**Supervisionado binário.** Carregue `fluxos_rede()`, prepare, separe 70/30 estratificado pelo tipo e compare "
     "uma `LogisticRegression` padronizada, uma `RandomForestClassifier(200)` e um `MLPClassifier((64, 32))`: AUC "
     "e o relatório por tipo da melhor.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from PKG.ids import preparar, relatorio_por_tipo, calibrar_limiar
fl = fluxos_rede(); X = preparar(fl); y = fl["ataque"]
X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(X, y, fl["tipo"], test_size=.3, random_state=0, stratify=fl["tipo"])
modelos = {"logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
           "random forest": RandomForestClassifier(200, random_state=0),
           "MLP": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=800, random_state=0))}
for nome, m in modelos.items():
    print(f"{nome:<14} AUC {roc_auc_score(y_te, m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]):.4f}")
print(relatorio_por_tipo(t_te, modelos["random forest"].predict(X_te)))'''),

    ("tarefa",
     "**Multiclasse.** Treine a Random Forest para prever o **tipo** de fluxo e mostre o `classification_report`. "
     "Qual ataque é mais confundido com o tráfego normal? Mostre as 6 características mais importantes.",
     r'''from sklearn.metrics import classification_report
rfm = RandomForestClassifier(200, random_state=0).fit(X_tr, t_tr)
print(classification_report(t_te, rfm.predict(X_te), digits=3))
print(pd.Series(rfm.feature_importances_, index=X.columns).sort_values(ascending=False).head(6).round(3))'''),

    ("tarefa",
     "**Detector de anomalias.** Treine um `IsolationForest(300)` **só com o tráfego normal** do treino. Use "
     "`-score_samples` como escore de anomalia e calibre o limiar para 1% de falsos alarmes nos normais do "
     "treino. Mostre o relatório por tipo no teste.",
     r'''iso = IsolationForest(n_estimators=300, random_state=0).fit(X_tr[y_tr == 0])
esc_tr, esc_te = -iso.score_samples(X_tr[y_tr == 0]), -iso.score_samples(X_te)
lim = calibrar_limiar(esc_tr, .01)
print("AUC anomalia:", round(roc_auc_score(y_te, esc_te), 4))
print(relatorio_por_tipo(t_te, esc_te > lim))'''),

    ("tarefa",
     "**O ataque novo (dia zero).** Retreine a Random Forest **sem nenhum exemplo de exfiltração** no treino (tire "
     "esse tipo do treino) e compare, no teste, a taxa de detecção da exfiltração pela RF e pelo Isolation Forest. "
     "Qual filosofia protege contra o desconhecido?",
     r'''sem_exf = t_tr != "exfiltração"
rf0 = RandomForestClassifier(200, random_state=0).fit(X_tr[sem_exf], y_tr[sem_exf])
exf = t_te == "exfiltração"
print("exfiltração detectada pela RF (nunca viu):", round(rf0.predict(X_te[exf]).mean(), 3))
print("exfiltração detectada pelo Isolation Forest:", round((esc_te[exf.to_numpy()] > lim).mean(), 3))'''),

    ("tarefa",
     "**Falsos alarmes por dia.** Suponha 2 milhões de fluxos normais por dia. Converta a taxa de falso alarme do "
     "RF e a do Isolation Forest (limiar de 1%) em alertas por dia. Quantos analistas, a 200 alertas por dia cada, "
     "cada abordagem exige? Combine as duas: alerta quando o RF **ou** a anomalia muito forte (limiar de 0,1%) "
     "disparam.",
     r'''normais = (t_te == "normal").to_numpy()
fa_rf = modelos["random forest"].predict(X_te)[normais].mean(); fa_iso = (esc_te[normais] > lim).mean()
lim_forte = calibrar_limiar(esc_tr, .001)
comb = (modelos["random forest"].predict(X_te) == 1) | (esc_te > lim_forte)
for nome, fa in [("RF", fa_rf), ("Isolation Forest", fa_iso), ("combinado", comb[normais].mean())]:
    print(f"{nome:<17} falso alarme {fa:.4f} -> {fa * 2e6:,.0f} alertas/dia -> {np.ceil(fa * 2e6 / 200):.0f} analistas")
print("combinado detecta exfiltração (com RF que viu):", round(comb[exf.to_numpy()].mean(), 3))'''),
]
