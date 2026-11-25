TITULO = "Phishing com foco no usuário"
MISSAO = (
    "O último capítulo do livro defende que um detector de phishing deve **ajudar o usuário a decidir**, não só "
    "acertar. Você vai:\n"
    "- treinar um classificador de e-mails (TF-IDF + regressão logística, com sinais do remetente e dos links);\n"
    "- explicar **cada alerta** com as palavras e os sinais que pesaram, numa mensagem que um funcionário entende;\n"
    "- testar a robustez contra o fraudador que troca palavras-gatilho por sinônimos."
)
ENTREGA = "o módulo `PKG/phishing.py` (as características do e-mail, o modelo, a explicação por e-mail em linguagem simples e o teste de troca de palavras)."
LEITURA = [("19.2 Background and Related Work", 328), ("19.3 The Dataset", 330),
           ("19.4 Understanding The Decision Behavior Of Machine Learning Models", 331), ("19.5 Designing the Artifact", 335)]

PASSOS = [
    ("texto",
     "## Ideia em 30 segundos\n"
     "Na regressão logística sobre TF-IDF, a contribuição de cada palavra para o escore é `peso × tf-idf`: a "
     "explicação é **exata**, sem aproximação. Os sinais fora do texto (domínio do remetente, número de links) "
     "entram como colunas extras.\n\n"
     "Uma boa mensagem ao usuário diz:\n"
     "- **o quê**: \"pede a sua senha\";\n"
     "- **por quê**: \"o domínio não é o da empresa\";\n"
     "- **o que fazer**: \"não clique, reporte ao TI\"."),

    ("arquivo",
     "**O módulo de phishing.** Crie `phishing.py` com:\n"
     "- `sinais(df, dominio=\"fortaleza-digital.com.br\")`: `dominio_externo` (o remetente **não termina** com "
     "`@` + o domínio) e `n_links`;\n"
     "- `Detector`:\n"
     "  - `fit(df)` ajusta um `TfidfVectorizer(ngram_range=(1, 2), min_df=2)` no texto e uma "
     "`LogisticRegression` em `[tfidf | sinais]`;\n"
     "  - `predict_proba(df)`;\n"
     "  - `explicar(linha, k=4)`: as k maiores contribuições positivas (palavras e sinais), com o valor;\n"
     "- `mensagem(prob, explicacao)`: o texto ao usuário.",
     "phishing.py",
     r'''"""Detector de phishing explicável, com foco no usuário."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SINAIS = ["dominio_externo", "n_links"]


def sinais(df, dominio="fortaleza-digital.com.br"):
    return pd.DataFrame({"dominio_externo": (~df["remetente"].str.endswith("@" + dominio)).astype(int),
                         "n_links": df["n_links"]}, index=df.index)


class Detector:
    def fit(self, df):
        self.vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        T = self.vet.fit_transform(df["texto"])
        self.modelo = LogisticRegression(max_iter=2000, C=5).fit(self._junta(T, df), df["phishing"])
        self.nomes = list(self.vet.get_feature_names_out()) + SINAIS
        return self

    def _junta(self, T, df):
        return hstack([T, csr_matrix(sinais(df).to_numpy(float))]).tocsr()

    def predict_proba(self, df):
        return self.modelo.predict_proba(self._junta(self.vet.transform(df["texto"]), df))[:, 1]

    def explicar(self, linha, k=4):
        x = self._junta(self.vet.transform([linha["texto"]]), linha.to_frame().T).toarray()[0]
        contrib = x * self.modelo.coef_[0]
        ordem = np.argsort(-contrib)[:k]
        return [(self.nomes[i], round(float(contrib[i]), 3)) for i in ordem if contrib[i] > 0]


def mensagem(prob, explicacao):
    if prob < 0.5:
        return f"Parece legítimo ({prob:.0%} de risco)."
    motivos = []
    for nome, _ in explicacao:
        if nome == "dominio_externo":
            motivos.append("o remetente é de fora da empresa")
        elif nome == "n_links":
            motivos.append("há links no e-mail")
        else:
            motivos.append(f'o texto contém "{nome}"')
    return (f"ATENÇÃO ({prob:.0%} de risco de phishing): " + "; ".join(motivos)
            + ". Não clique em links nem informe senhas. Reporte ao TI.")'''),

    ("tarefa",
     "**Treine e avalie.** Carregue `emails()`, separe 70/30 estratificado, treine o `Detector` e mostre a AUC, a "
     "precisão e a revocação com limiar 0,5. Compare com um modelo **só de texto** (sem os sinais) para ver o "
     "quanto o remetente ajuda.",
     r'''from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PKG.phishing import Detector, mensagem
em = emails()
tr, te = train_test_split(em, test_size=.3, random_state=0, stratify=em["phishing"])
det = Detector().fit(tr); p = det.predict_proba(te)
print(f"completo: AUC {roc_auc_score(te.phishing, p):.4f} | precisão {precision_score(te.phishing, p >= .5):.3f} | revocação {recall_score(te.phishing, p >= .5):.3f}")
vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(tr.texto)
so_texto = LogisticRegression(max_iter=2000).fit(vet.transform(tr.texto), tr.phishing)
print("só texto: AUC", round(roc_auc_score(te.phishing, so_texto.predict_proba(vet.transform(te.texto))[:, 1]), 4))'''),

    ("tarefa",
     "**Explicar ao usuário.** Para 3 phishings e 1 e-mail legítimo do teste, mostre o texto, a explicação e a "
     "`mensagem` que o funcionário veria.",
     r'''amostras = pd.concat([te[te.phishing == 1].head(3), te[te.phishing == 0].head(1)])
for _, linha in amostras.iterrows():
    pr = det.predict_proba(linha.to_frame().T)[0]
    print(f"[{linha.remetente}] {linha.texto}")
    print("   ->", mensagem(pr, det.explicar(linha))); print()'''),

    ("tarefa",
     "**O que o modelo aprendeu.** Mostre os 10 termos com maior peso para phishing e os 10 com maior peso para "
     "legítimo. Algum deles é um atalho perigoso, ou seja, uma palavra que aparece por acaso e não diz nada sobre "
     "golpe?",
     r'''coefs = pd.Series(det.modelo.coef_[0], index=det.nomes).sort_values()
print("phishing:", coefs.tail(10).index.tolist()[::-1]); print("legítimo:", coefs.head(10).index.tolist())'''),

    ("tarefa",
     "**O fraudador troca palavras.** Pegue os phishings do teste e troque as palavras-gatilho por sinônimos (por "
     "exemplo, senha → credencial, clique → acesse, urgente → importante, link → endereço, conta → cadastro). Qual "
     "fração ainda é detectada pelo modelo completo e pelo só de texto? Por que os sinais do remetente tornam a "
     "evasão mais cara?",
     r'''trocas = {"senha": "credencial", "clique": "acesse", "urgente": "importante", "link": "endereço", "conta": "perfil",
          "bloqueada": "suspensa", "cartão": "plástico", "token": "código", "prêmio": "brinde", "verificar": "confirmar",
          "atualize": "renove", "imediatamente": "já"}
original = te[te.phishing == 1]; ph = original.copy()
for a, b_ in trocas.items():
    ph["texto"] = ph["texto"].str.replace(rf"\b{a}\b", b_, case=False, regex=True)
taxa = lambda p: f"{np.mean(p >= .5):.0%}"
print("completo:  antes", taxa(det.predict_proba(original)), "| depois", taxa(det.predict_proba(ph)))
print("só texto:  antes", taxa(so_texto.predict_proba(vet.transform(original.texto))[:, 1]),
      "| depois", taxa(so_texto.predict_proba(vet.transform(ph.texto))[:, 1]))
# Trocar palavras é de graça para o fraudador; trocar o domínio do remetente e tirar os links custa caro.'''),
]
