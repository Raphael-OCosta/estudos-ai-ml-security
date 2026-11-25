# Estudos — *AI, Machine Learning and Deep Learning: A Security Perspective* (Fei Hu e Xiali Hei, orgs.) · trilha prática

> **A melhor forma de estudar um livro técnico é ler e praticar ao mesmo tempo.**
> Aqui a prática vem primeiro. Cada capítulo é uma **missão** curta: você lê só o essencial, ataca o próprio modelo e depois o defende.

O livro reúne 19 capítulos sobre a segurança **da** IA (ataques e defesas) e a IA **para** segurança (detecção de intrusão, fraude e phishing). Nas 10 missões, você trabalha no **laboratório de segurança de IA da Fortaleza Digital** (fictícia), sempre com modelos e dados próprios.

## A trilha (51 passos)

| Cap. | Missão | Você constrói |
|---|---|---|
| 1 | Exemplos adversariais | modelo de ameaça, **FGSM e PGD**, curva de robustez, transferência, caixa preta por consultas, `ataques.py` |
| 2 | Envenenamento | inversão de rótulos aleatória × direcionada, **backdoor com gatilho**, evasão do SVM linear, `envenenamento.py` |
| 3 | Federado | **FedAvg do zero**, cliente malicioso, mediana e média aparada, detecção, inferência de pertencimento, `federado.py` |
| 4 | Reforço sob ataque | Q-learning num armazém, ataque às observações, recompensas envenenadas, fusão de sensores, `rl_seguro.py` |
| 5 | Redes bayesianas | ameaças a IoT, inferência por enumeração, diagnóstico, ranking de mitigação, `bayes_ameacas.py` |
| 6 | Defesas | **treino adversarial min-max**, compressão, detector, saneamento contra envenenamento, `defesas.py` |
| 7 | Robustez certificada | raio exato do linear, **suavização aleatória** (Cohen et al.), acurácia certificada, `certificado.py` |
| 8 | Detecção de intrusão | IIoT com 4 ataques, supervisionado × Isolation Forest, ataque de dia zero, alertas por dia, `ids.py` |
| 9 | Aplicações | inundação de interesses em NDN, **red team do antifraude** e defesa, `aplicacoes.py` |
| 10 | Phishing | TF-IDF + sinais, **explicação ao usuário**, troca de palavras pelo fraudador, `phishing.py` |

## Diferenças em relação ao livro

- O livro é uma coletânea de surveys. A trilha transforma os temas em experimentos reprodutíveis com **dados fictícios** (`comum/dados.py`) e os dígitos do scikit-learn.
- As GANs do cap. 17 viraram uma busca de evasão de caixa preta, a mesma ideia (gerar fraudes que passam) com menos peso. O deep RL virou Q-learning tabular, para os ataques ficarem visíveis.
- Todos os ataques são feitos contra modelos que você mesmo treina, no papel de *red team*.
- As páginas indicadas nos cadernos são as do PDF em inglês. A leitura é **opcional** e só do essencial.

## Como usar

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-estudos.txt
.venv\Scripts\python.exe ferramentas\testar_cadernos.py   # opcional: roda o gabarito (0 erros)
```

Abra `praticar/cap01_*.ipynb` no VS Code e escolha o kernel `.venv`. O [Guia de Estudos (PDF)](Guia_de_Estudos.pdf) mostra o mapa e a rotina de 60 minutos.

## Direitos autorais

O livro **não** está neste repositório. Use a sua própria cópia.
As missões, os cadernos, os dados fictícios, os scripts e o guia são autorais, sob licença [MIT](LICENSE).
