# README pentru redactarea subcapitolului 4.1

## Subcapitol vizat

**4.1. Strategia de testare si configurarea experimentelor**

Acest fisier contine contextul, ideile principale, datele concrete din proiect si fragmentele de cod relevante pentru redactarea subcapitolului 4.1 din lucrarea de licenta. Textul poate fi folosit ca material de lucru intr-un chat sau ca suport pentru integrarea subcapitolului in documentul final.

## Contextul lucrarii

Lucrarea are ca tema dezvoltarea unei solutii bazate pe inteligenta artificiala pentru clasificarea imaginilor dermatoscopice ale leziunilor cutanate. Problema finala este formulata ca o clasificare binara intre clasele:

- `benign`
- `malignant`

Capitolul 3 a prezentat implementarea pipeline-ului de preprocesare, antrenare si predictie. Capitolul 4 se concentreaza pe testarea modelelor si pe analiza rezultatelor obtinute. Subcapitolul 4.1 trebuie sa explice modul in care au fost configurate experimentele, ce date au fost folosite pentru evaluare, ce metrici au fost calculate si cum au fost comparate modelele.

## Rolul subcapitolului in lucrare

Subcapitolul 4.1 trebuie sa prezinte:

- separarea dintre antrenare, validare si testare;
- configurarea comuna a experimentelor;
- modul de incarcare a imaginilor pentru evaluare;
- faptul ca setul de test nu este augmentat;
- pragurile de decizie folosite pentru fiecare model;
- metricile utilizate pentru evaluare;
- tipurile de experimente realizate;
- modul de comparare intre EfficientNetB0 si ResNet50;
- metoda de output fusion;
- organizarea fisierelor generate de pipeline.

Subcapitolul nu trebuie sa intre in interpretarea detaliata a rezultatelor finale. Interpretarea extinsa poate fi tratata in subcapitolele urmatoare din Capitolul 4.

## Fisiere relevante din proiect

Implementarea strategiei de testare si evaluare se afla in:

```text
pipeline_rebuilt/config.py
pipeline_rebuilt/common.py
pipeline_rebuilt/evaluate_model.py
pipeline_rebuilt/compare_models.py
pipeline_rebuilt/roc_curve_models.py
pipeline_rebuilt/output_fusion.py
pipeline_rebuilt/external_validation.py
pipeline_rebuilt/model_registry.py
```

Rezultatele generate sunt salvate in:

```text
pipeline_rebuilt_outputs/models/
pipeline_rebuilt_outputs/plots/
pipeline_rebuilt_outputs/confusion_matrix/
pipeline_rebuilt_outputs/reports/
```

Documentul Word generat pentru acest subcapitol este:

```text
Subcapitol_4_1_Strategia_Testare_Configurare_Experimente.docx
```

## Configurarea comuna a experimentelor

Valorile principale sunt definite in `pipeline_rebuilt/config.py`:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
```

Semnificatie:

- `IMG_SIZE = (224, 224)` - toate imaginile sunt redimensionate la 224x224 pixeli;
- `BATCH_SIZE = 16` - imaginile sunt evaluate in loturi de cate 16;
- `SEED = 42` - seed fix pentru reproductibilitate;
- `validation_split = 0.2` - 20% din setul de antrenare este folosit pentru validare interna.

Aceeasi configurare este folosita pentru ambele arhitecturi evaluate:

- EfficientNetB0;
- ResNet50.

## Structura datasetului folosit in experimente

Datasetul reorganizat este structurat astfel:

```text
dataset/new_dataset/
  train/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

Distributia imaginilor este:

| Subset | Benign | Malignant | Total |
|---|---:|---:|---:|
| Train | 1440 | 1197 | 2637 |
| Test | 359 | 300 | 659 |

Observatii importante:

- setul de test este separat de antrenare;
- setul de test nu este folosit pentru ajustarea ponderilor modelului;
- validarea interna este extrasa din setul de train;
- evaluarea finala se face pe cele 659 de imagini din setul de test.

## Generatorul pentru testare

Pentru evaluare se foloseste un generator fara augmentare:

```python
def create_eval_generator(data_dir, preprocess_function, batch_size=BATCH_SIZE):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_function)
    return datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
```

Explicatii:

- `target_size=IMG_SIZE` redimensioneaza imaginile la 224x224 pixeli;
- `class_mode="binary"` este potrivit pentru cele doua clase;
- `shuffle=False` pastreaza ordinea imaginilor;
- nu se aplica rotatii, zoom, flip sau modificari de luminozitate;
- se aplica doar preprocesarea specifica modelului.

Setul de test nu este augmentat deoarece scopul evaluarii este masurarea comportamentului modelului pe imagini cat mai apropiate de datele reale de intrare.

## Reproductibilitatea experimentelor

In `common.py`, reproductibilitatea este sustinuta prin functia:

```python
def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
```

Aceasta seteaza seed-ul pentru:

- biblioteca `random`;
- NumPy;
- TensorFlow;
- Keras.

Prin aceasta abordare, impartirea datelor, initializarile si anumite operatii ale pipeline-ului devin mai stabile de la o rulare la alta.

## Modelele evaluate

Modelele sunt definite in `pipeline_rebuilt/model_registry.py`.

Pentru EfficientNetB0:

```python
threshold=EFFICIENTNET_THRESHOLD
last_conv_layer="top_activation"
```

Pentru ResNet50:

```python
threshold=RESNET_THRESHOLD
last_conv_layer="conv5_block3_out"
```

Pragurile finale definite in `config.py` sunt:

```python
EFFICIENTNET_THRESHOLD = 0.40
RESNET_THRESHOLD = 0.60
OUTPUT_FUSION_THRESHOLD = 0.50
```

Interpretare:

- EfficientNetB0 foloseste pragul `0.40`;
- ResNet50 foloseste pragul `0.60`;
- output fusion foloseste pragul `0.50`.

## Pragurile de decizie

Modelele produc o probabilitate pentru clasa `malignant`. Eticheta finala se obtine prin compararea probabilitatii cu un prag:

```python
y_pred = (probs >= threshold).astype(int)
```

In `evaluate_model.py`, pragul poate fi cautat pe intervalul:

```python
thresholds = np.arange(0.35, 0.81, 0.05)
```

Criteriul folosit este `balanced accuracy`.

Formula folosita in cod:

```python
balanced_acc = (recall_benign + recall_malignant) / 2
```

Aceasta metoda este utila deoarece:

- nu favorizeaza direct clasa majoritara;
- tine cont de performanta pe ambele clase;
- permite observarea compromisului dintre fals pozitive si fals negative.

## Metricile folosite

Pentru evaluare sunt folosite urmatoarele metrici:

| Metrica | Rol |
|---|---|
| Accuracy | Proportia totala a predictiilor corecte |
| Precision | Cate predictii `malignant` sunt corecte |
| Recall | Cate cazuri `malignant` sunt detectate |
| F1-score | Media armonica intre precision si recall |
| Balanced Accuracy | Media performantelor pe cele doua clase |
| Matrice de confuzie | Evidentiaza TN, FP, FN si TP |
| Curba ROC | Analizeaza separarea claselor independent de un singur prag |
| AUC | Rezuma performanta curbei ROC |

In context medical, `recall` pentru clasa `malignant` si numarul de fals negative sunt foarte importante, deoarece un fals negativ inseamna o leziune maligna clasificata ca benigna.

## Calcularea metricilor in cod

Fragment relevant din `compare_models.py`:

```python
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
balanced_acc = ((tp / (tp + fn + 1e-8)) + (tn / (tn + fp + 1e-8))) / 2
```

Rezultatele sunt salvate intr-un fisier CSV:

```text
pipeline_rebuilt_outputs/reports/model_comparison_rebuilt.csv
```

## Experimente realizate

Experimentele principale sunt:

1. Evaluarea individuala a modelului EfficientNetB0.
2. Evaluarea individuala a modelului ResNet50.
3. Compararea modelelor intr-un tabel comun.
4. Generarea curbelor ROC.
5. Aplicarea metodei de output fusion.

### 1. Evaluarea EfficientNetB0

Script relevant:

```text
pipeline_rebuilt/evaluate_model.py
```

Output-uri relevante:

```text
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_new_best_model_v3_rebuilt.png
pipeline_rebuilt_outputs/plots/training_plot_new_best_model_v3_rebuilt.png
```

### 2. Evaluarea ResNet50

Script relevant:

```text
pipeline_rebuilt/evaluate_model.py
```

Output-uri relevante:

```text
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_model_2_resnet50_rebuilt.png
pipeline_rebuilt_outputs/plots/training_plot_model_2_resnet50_rebuilt.png
```

### 3. Compararea modelelor

Script relevant:

```text
pipeline_rebuilt/compare_models.py
```

Output-uri relevante:

```text
pipeline_rebuilt_outputs/reports/model_comparison_rebuilt.csv
pipeline_rebuilt_outputs/plots/model_comparison_rebuilt.png
```

Rezultatele numerice obtinute:

| Model | Prag | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB0 Rebuilt | 0.40 | 0.8498 | 0.7947 | 0.9033 | 0.8456 | 0.8542 | 289 | 70 | 29 | 271 |
| ResNet50 Rebuilt | 0.60 | 0.8756 | 0.8759 | 0.8467 | 0.8610 | 0.8732 | 323 | 36 | 46 | 254 |

Interpretare scurta:

- EfficientNetB0 are recall mai mare pentru clasa `malignant`;
- ResNet50 are accuracy si precision mai mari;
- EfficientNetB0 produce mai putine fals negative;
- ResNet50 produce mai putine fals pozitive;
- modelele au comportamente complementare, ceea ce justifica testarea unei metode de output fusion.

### 4. Curbele ROC

Script relevant:

```text
pipeline_rebuilt/roc_curve_models.py
```

Output relevant:

```text
pipeline_rebuilt_outputs/plots/roc_curve_models_rebuilt.png
```

Curbele ROC sunt utile deoarece evalueaza capacitatea modelelor de a separa clasele pentru mai multe praguri posibile, nu doar pentru pragul final ales.

### 5. Output fusion

Script relevant:

```text
pipeline_rebuilt/output_fusion.py
```

Output-uri relevante:

```text
pipeline_rebuilt_outputs/reports/output_fusion_report.csv
pipeline_rebuilt_outputs/reports/output_fusion_probabilities.csv
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_output_fusion.png
```

Metoda folosita combina probabilitatile produse de cele doua modele printr-o medie ponderata.

Fragment relevant:

```python
weighted_probs += weight * two_class_probs
y_pred = (weighted_probs[:, 1] >= threshold).astype(int)
```

Ponderile sunt calculate pe baza acuratetii modelelor pe setul de antrenare:

```python
train_accuracies = np.array([model_train_accuracy(spec) for spec in specs], dtype="float32")
weights = train_accuracies / np.sum(train_accuracies)
```

Rezumat output fusion:

| Clasa | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| benign | 0.8947 | 0.8524 | 0.8730 | 359 |
| malignant | 0.8328 | 0.8800 | 0.8558 | 300 |
| accuracy | 0.8649 | 0.8649 | 0.8649 | 659 |

Interpretare scurta:

- output fusion obtine accuracy aproximativ `0.8649`;
- recall-ul pentru `malignant` este `0.88`;
- metoda combina predictiile celor doua modele, dar nu depaseste automat fiecare model la toate metricile;
- utilitatea metodei trebuie analizata in functie de obiectivul aplicatiei.

## Figuri recomandate pentru subcapitol

Pentru subcapitolul 4.1 pot fi incluse urmatoarele figuri:

```text
pipeline_rebuilt_outputs/plots/model_comparison_rebuilt.png
pipeline_rebuilt_outputs/plots/roc_curve_models_rebuilt.png
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_new_best_model_v3_rebuilt.png
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_model_2_resnet50_rebuilt.png
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_output_fusion.png
```

Legende posibile:

- Figura 4.1. Compararea modelelor EfficientNetB0 si ResNet50 pe setul de test.
- Figura 4.2. Curbele ROC pentru modelele evaluate.
- Figura 4.3. Matricea de confuzie pentru EfficientNetB0.
- Figura 4.4. Matricea de confuzie pentru ResNet50.
- Figura 4.5. Matricea de confuzie pentru metoda de output fusion.

## Structura recomandata pentru redactare

Subcapitolul poate fi structurat astfel:

1. Introducere despre scopul testarii.
2. Explicarea separarii train/validation/test.
3. Prezentarea configurarii comune: imagine 224x224, batch 16, seed 42.
4. Explicarea generatorului de testare fara augmentare.
5. Prezentarea modelelor evaluate.
6. Explicarea pragurilor de decizie.
7. Prezentarea metricilor folosite.
8. Descrierea experimentelor realizate.
9. Prezentarea comparatiei intre EfficientNetB0 si ResNet50.
10. Descrierea metodei de output fusion.
11. Concluzie despre reproductibilitate si relevanta strategiei de testare.

## Idei principale pentru redactare

Idei care trebuie transmise in text:

- testarea este separata de antrenare;
- setul de test nu este folosit pentru optimizarea modelelor;
- augmentarea este folosita la antrenare, nu la testare;
- modelele sunt evaluate in conditii identice;
- pragurile de decizie sunt explicite si influenteaza rezultatele;
- balanced accuracy este utila deoarece tine cont de ambele clase;
- matricea de confuzie este esentiala pentru analiza erorilor;
- EfficientNetB0 si ResNet50 au comportamente diferite;
- output fusion verifica daca o decizie combinata poate imbunatati echilibrul rezultatelor;
- fisierele de output sunt organizate astfel incat experimentele sa fie reproductibile.

## Formulare de introducere posibila

```text
Evaluarea modelelor dezvoltate in cadrul proiectului a fost organizata ca o etapa separata fata de antrenare, astfel incat performanta raportata sa reflecte comportamentul pe imagini care nu au fost folosite pentru optimizarea parametrilor. Strategia de testare urmareste masurarea performantei individuale pentru fiecare arhitectura CNN, compararea modelelor in conditii identice si verificarea unei variante de decizie combinate prin output fusion.
```

## Formulare de concluzie posibila

```text
Strategia de testare folosita in proiect separa clar antrenarea de evaluarea finala si foloseste aceleasi conditii pentru toate modelele comparate. Prin pastrarea setului de test fara augmentare, folosirea unor praguri explicite, raportarea metricilor complementare si salvarea rezultatelor in fisiere separate, experimentele devin reproductibile si interpretatibile. Aceasta configurare este adecvata pentru clasificarea imaginilor dermatoscopice, deoarece pune accent nu doar pe performanta globala, ci si pe tipurile de erori care pot influenta utilitatea practica a sistemului.
```

