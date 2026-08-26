# Stroke Prediction – Machine Learning

## Descriere

Acest proiect urmărește dezvoltarea și evaluarea unor modele de **Machine Learning** pentru estimarea riscului de accident vascular cerebral (**stroke**) pe baza unor caracteristici demografice și medicale.

Un aspect important al problemei este dezechilibrul dintre clase, deoarece numărul cazurilor pozitive este mult mai mic decât cel al cazurilor negative. Din acest motiv, evaluarea modelelor s-a bazat în principal pe:

- **Recall**
- **F1-score**
- **False Negatives (FN)**
- **False Positives (FP)**

Obiectivul principal este identificarea unui număr cât mai mare de cazuri pozitive și, în special, reducerea numărului de cazuri de stroke care sunt omise de model.

---

## Dataset

Pentru proiect a fost utilizat datasetul `healthcare-dataset-stroke-data.csv`.

**Sursă:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Variabila țintă este:

- `stroke = 0` – cazul nu este asociat unui stroke;
- `stroke = 1` – cazul este asociat unui stroke.

Coloana `id` nu este utilizată pentru predicție, deoarece reprezintă doar identificatorul observației.

---

## Explorarea datelor

În prima etapă a fost analizată structura datasetului și au fost calculate statistici descriptive folosind:

- `info()`
- `describe()`
- `head()`

Au fost verificate și valorile lipsă, cu o atenție deosebită asupra coloanei `bmi`.

De asemenea, au fost identificate variabilele categorice și valorile distincte ale acestora.

### Distribuția variabilei țintă

Distribuția variabilei `stroke` a fost analizată atât numeric, cât și grafic.

Această analiză a evidențiat dezechilibrul puternic dintre clase, ceea ce a reprezentat un aspect important în alegerea și evaluarea modelelor.

Au fost analizate și relațiile dintre variabilele categorice și `stroke`, precum și distribuția variabilelor numerice `avg_glucose_level` și `bmi` prin intermediul boxplot-urilor.

---

## Analiza suplimentară a datelor

Au fost realizate mai multe analize pentru identificarea unor posibile relații între variabile și apariția unui stroke.

### Vârsta

A fost verificată distribuția variabilei `age`, inclusiv existența unor observații cu vârsta sub 18 ani și distribuția cazurilor de stroke în acest grup.

### Hipertensiune și boli cardiace

A fost analizată interacțiunea dintre `hypertension` și `heart_disease`, urmărindu-se numărul total de observații și numărul de cazuri de stroke pentru fiecare combinație.

### Nivelul glicemiei

A fost analizată variabila `avg_glucose_level`, folosind pragul de `126` pentru separarea valorilor ridicate de celelalte valori.

Pe baza acestei analize a fost creată ulterior o caracteristică nouă:

```python
high_glucose = (avg_glucose_level >= 126).astype(int)
```

Această caracteristică a fost utilizată într-un experiment suplimentar de **Feature Engineering**.

---

## Preprocesarea datelor

Datele au fost împărțite astfel:

- **70%** – set de antrenare;
- **30%** – set de testare.

Pentru păstrarea proporției dintre clase a fost utilizat `stratify=y`, iar `random_state=42` a fost folosit pentru reproductibilitate.

Variabilele categorice au fost transformate folosind `OneHotEncoder`, cu `handle_unknown="ignore"`.

Valorile lipsă din `bmi` au fost înlocuite cu mediana coloanei.

În experimentul care a inclus feature engineering pentru glicemie, variabilele numerice au fost și standardizate folosind `StandardScaler`.

Toate aceste operații au fost integrate în `Pipeline` și `ColumnTransformer`.

---

## Modele utilizate

Au fost analizate mai multe abordări de clasificare.

### 1. Logistic Regression

Primul model a fost o regresie logistică standard.

### 2. Logistic Regression + `class_weight`

Pentru a trata dezechilibrul dintre clase a fost utilizată configurația:

```python
class_weight="balanced"
```

Această abordare acordă o importanță mai mare clasei minoritare.

### 3. Random Forest

A fost utilizat un model Random Forest cu:

- `n_estimators=200`
- `random_state=42`
- `class_weight="balanced"`

### 4. Random Forest + GridSearchCV

Pentru Random Forest au fost căutați automat cei mai potriviți hiperparametri folosind `GridSearchCV` și `StratifiedKFold` cu 5 fold-uri.

Căutarea a inclus:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`

Criteriul de selecție a fost **F1-score**.

### 5. XGBoost

A fost testat și XGBoost cu parametri precum:

- `n_estimators=200`
- `max_depth=5`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`

### 6. XGBoost + `scale_pos_weight`

Pentru gestionarea dezechilibrului claselor a fost calculat raportul dintre numărul cazurilor negative și cel al cazurilor pozitive din setul de antrenare și utilizat pentru `scale_pos_weight`.

### 7. XGBoost + GridSearchCV

Pentru XGBoost au fost optimizați:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `scale_pos_weight`

Și în acest caz selecția s-a realizat pe baza **F1-score**, folosind 5-fold Stratified Cross-Validation.

---

## Feature Engineering

A fost realizat un experiment suplimentar în care a fost introdusă variabila `high_glucose`.

Aceasta indică dacă nivelul glicemiei este cel puțin 126:

```python
high_glucose = (avg_glucose_level >= 126).astype(int)
```

Modelul utilizat în acest experiment a fost **Logistic Regression** cu:

- `class_weight="balanced"`
- `StandardScaler`
- noua caracteristică `high_glucose`

Performanța acestuia a fost comparată cu modelele de Logistic Regression anterioare folosind:

- Recall
- F1-score
- FN
- FP

---

## AutoML

Pentru a compara modelele construite manual cu o abordare automată, a fost utilizat **FLAML AutoML**.

Configurația utilizată a fost:

- `task: classification`
- `metric: f1`
- `time_budget=120` secunde
- `seed=42`

Au fost permise mai multe tipuri de algoritmi:

- LightGBM
- Random Forest
- XGBoost
- XGBoost cu limitarea adâncimii
- Extra Trees
- HistGradientBoosting

FLAML a selectat automat modelul și configurația cu cea mai bună performanță în timpul alocat.

---

## Evaluarea modelelor

Modelele au fost evaluate folosind mai multe metrici.

### Recall

Recall-ul măsoară proporția cazurilor pozitive reale care sunt identificate corect de model.

În acest proiect, Recall este deosebit de important deoarece un **False Negative** reprezintă un caz pozitiv pe care modelul nu îl identifică.

### F1-score

F1-score combină **Precision** și **Recall** într-o singură metrică și oferă un compromis între cele două.

### Confusion Matrix

Pentru fiecare model au fost analizate:

- **TN** – True Negative
- **TP** – True Positive
- **FN** – False Negative
- **FP** – False Positive

---

## Alegerea modelului final

Experimentarea a evidențiat importanța gestionării dezechilibrului dintre clase.

Modelele standard au avut dificultăți în identificarea clasei minoritare, în timp ce utilizarea unor tehnici precum `class_weight="balanced"` și `scale_pos_weight` a îmbunătățit identificarea cazurilor pozitive.

Dintre variantele analizate, **Logistic Regression cu `class_weight="balanced"`** a fost selectată ca model final, deoarece a obținut:

- **Recall = 0.79**
- **FN = 16**

Alegerea nu s-a bazat exclusiv pe F1-score. Deoarece obiectivul proiectului este identificarea unui număr cât mai mare de cazuri pozitive, **Recall** și numărul de **False Negatives** au fost considerate prioritare.

---

## Precision-Recall Curve

Pentru modelul final au fost calculate probabilitățile de apartenență la clasa pozitivă și a fost construită curba **Precision-Recall**.

Aceasta permite analizarea compromisului dintre Precision și Recall pentru diferite praguri de clasificare.

A fost calculată și valoarea **PR AUC**, care oferă o măsură sintetică a performanței modelului pe întregul interval de praguri.

---

## Analiza pragului de clasificare

Pragul implicit de `0.5` nu este neapărat cel mai potrivit pentru această problemă.

Au fost testate următoarele praguri:

`0.2`, `0.3`, `0.4`, `0.5`, `0.6`, `0.7`

| Threshold | Precision | Recall | F1 | FN | FP |
|----------:|----------:|-------:|---:|---:|---:|
| 0.2 | 0.09 | 0.91 | 0.16 | 7 | 727 |
| 0.3 | 0.10 | 0.87 | 0.18 | 10 | 599 |
| 0.4 | 0.11 | 0.83 | 0.20 | 13 | 492 |
| 0.5 | 0.13 | 0.79 | 0.22 | 16 | 398 |
| 0.6 | 0.16 | 0.76 | 0.26 | 18 | 308 |
| 0.7 | 0.19 | 0.69 | 0.30 | 23 | 221 |

Pe măsură ce crește pragul:

- Precision se îmbunătățește;
- numărul de False Positive scade;
- Recall scade;
- numărul de False Negative crește.

De exemplu, la pragul `0.2`, modelul obține un **Recall de 0.91** și doar **7 FN**, dar produce **727 FP**.

La pragul `0.7`, numărul de FP scade la **221**, însă FN crește la **23**, iar Recall scade la **0.69**.

Prin urmare, alegerea pragului depinde de obiectivul aplicației. În cazul acestui proiect, unde este important să fie evitate cazurile de stroke omise, un prag mai scăzut poate fi preferabil chiar dacă generează mai multe False Positive.

---

## Concluzii

În cadrul proiectului au fost analizați mai mulți algoritmi de clasificare pentru estimarea riscului de accident vascular cerebral.

Experimentarea a evidențiat importanța gestionării dezechilibrului dintre clase. Utilizarea unor tehnici precum `class_weight="balanced"` și `scale_pos_weight` a contribuit la îmbunătățirea identificării cazurilor din clasa pozitivă.

**Logistic Regression cu `class_weight="balanced"`** a fost selectată ca model final, obținând un **Recall de 0.79** și **16 False Negatives**.

Deși unele modele au obținut un F1-score mai ridicat, Logistic Regression cu `class_weight="balanced"` a fost preferată deoarece obiectivul principal al proiectului este identificarea unui număr cât mai mare de cazuri pozitive și reducerea cazurilor de AVC omise de model.

Analiza pragurilor a demonstrat, de asemenea, că performanța modelului poate fi modificată semnificativ prin ajustarea pragului de clasificare. Un prag mai mic favorizează Recall-ul și reduce numărul de False Negative, în timp ce un prag mai mare favorizează Precision-ul și reduce False Positive.

În concluzie, rezultatele demonstrează că atât alegerea algoritmului, cât și gestionarea dezechilibrului dintre clase și alegerea pragului de clasificare au un impact important asupra performanței unui model de clasificare.

---

## Tehnologii utilizate

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- FLAML
- MLflow
- Google Colab

---

## Rulare

Proiectul a fost dezvoltat și testat în **Google Colab**.

Datasetul este încărcat în notebook folosind:

```python
from google.colab import files

uploaded = files.upload()
```

După încărcare, datasetul este citit folosind Pandas:

```python
import pandas as pd

db = pd.read_csv("/content/healthcare-dataset-stroke-data.csv")
```

Pentru rularea completă a notebook-ului este necesară instalarea bibliotecilor utilizate, inclusiv `xgboost` și `flaml`.

```bash
pip install xgboost flaml
```

După instalarea dependențelor și încărcarea datasetului, notebook-ul poate fi rulat de la început până la final pentru reproducerea experimentelor și a rezultatelor prezentate mai sus.
