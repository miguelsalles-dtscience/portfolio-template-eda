# Template EDA Profissional

> Análise Exploratória de Dados com rigor estatístico — do CSV bruto ao relatório HTML em minutos.

**Autor:** Miguel Salles Reis · [github.com/miguelsalles-dtscience](https://github.com/miguelsalles-dtscience)

---

## O que é este template

Um notebook Jupyter estruturado para conduzir uma EDA completa em qualquer dataset tabular. Não é um wrapper do pandas-profiling — cada seção foi construída do zero com lógica estatística explícita e comentada, pensada para quem quer entender o que está rodando, não só colar e executar.

O entregável final é um **relatório HTML autocontido** — sem dependências externas, pronto para enviar ao cliente ou anexar num portfólio.

---

## O que está incluído

| Seção | O que entrega |
|---|---|
| 0 · Configuração | Imports, paleta de cores, funções auxiliares |
| 1 · Carregamento | Leitura do CSV com diagnóstico inicial e fallback para dataset demo |
| 2 · Data Quality Score | Nota objetiva de 0–100 em 5 dimensões de qualidade |
| 3 · Limpeza Automática | Duplicatas, nulos, tipos, winsorização — com log de cada ação |
| 4 · EDA Univariada | Distribuições, skewness, curtose, boxplots |
| 5 · EDA Bivariada | Correlações de Pearson com p-valor e tamanho de efeito (Cohen) |
| 6 · Testes de Hipótese | Seleção automática entre t-test / Mann-Whitney / ANOVA / Kruskal-Wallis |
| 7 · Detecção de Outliers | IQR + Z-score + Isolation Forest com consenso por votação |
| 8 · Relatório HTML | Exportação com narrativa interpretativa automática |

---

## Diferenciais

**Data Quality Score** — em vez de listar problemas subjetivamente, o template calcula uma nota ponderada em 5 dimensões (completude, unicidade, consistência, validade de tipos, cardinalidade). Você sabe exatamente o quanto o dataset melhorou após a limpeza.

**Testes de hipótese com seleção automática** — o template verifica a normalidade de cada variável (Shapiro-Wilk) e escolhe o teste correto para a comparação de grupos. Paramétrico ou não-paramétrico, 2 grupos ou múltiplos — sem intervenção manual.

**Outliers por consenso** — um ponto só é marcado como outlier confirmado se pelo menos 2 dos 3 métodos concordam. Reduz falsos positivos e torna a decisão auditável.

**Relatório HTML autocontido** — os gráficos são embutidos em base64. Um único arquivo `.html` que abre em qualquer navegador, sem precisar de servidor, Python ou Jupyter instalado.

**Código comentado para aprender** — cada bloco explica o *porquê* da decisão estatística, não só o *como*. Útil para estudos, entrevistas e documentação de projetos.

---

## Como usar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar o dataset

Abra o notebook e altere a variável na Seção 1:

```python
CSV_PATH = "seu_dataset.csv"   # ← altere aqui
```

Caso o arquivo não seja encontrado, o template carrega automaticamente um dataset de RH fictício com problemas intencionais (nulos, outliers, duplicatas, tipos errados) para demonstração.

### 3. Executar

```
Kernel → Restart & Run All
```

### 4. Coletar os resultados

Os arquivos são salvos na pasta `outputs/`:

| Arquivo | Conteúdo |
|---|---|
| `01_dq_score.png` | Gráfico do Data Quality Score |
| `02_univariada.png` | Distribuições e boxplots |
| `03_correlacoes.png` | Heatmap de correlações + significância |
| `04_outliers.png` | Comparação dos métodos de detecção |
| `relatorio_eda.html` | Relatório completo — abra no navegador |

---

## Requisitos

- Python 3.9+
- Jupyter Notebook ou JupyterLab

Dependências completas no `requirements.txt`. As principais: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`.

---

## Licença

Uso pessoal e comercial permitido. Redistribuição ou revenda do template original não é permitida.

---

*Template EDA Profissional · Miguel Salles Reis · [github.com/miguelsalles-dtscience](https://github.com/miguelsalles-dtscience)*
