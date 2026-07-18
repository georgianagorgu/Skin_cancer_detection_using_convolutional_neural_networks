# README pentru redactarea subcapitolului 4.2

## Subcapitol vizat

**4.2. Evaluarea modelelor individuale pe setul de test**

Acest fisier contine informatiile esentiale pentru redactarea subcapitolului 4.2. Spre deosebire de subcapitolul 4.1, aici nu trebuie reluata strategia generala de testare, ci trebuie prezentate rezultatele efective obtinute de fiecare model pe setul de test.

## Rolul subcapitolului

Subcapitolul 4.2 trebuie sa arate cum se comporta separat cele doua modele antrenate:

- EfficientNetB0;
- ResNet50.

Accentul cade pe:

- valorile metricilor obtinute pe setul de test;
- interpretarea matricilor de confuzie;
- diferenta dintre fals pozitive si fals negative;
- compararea directa a celor doua modele;
- concluzia privind modelul mai sensibil si modelul mai precis.

## Fisiere relevante

Rezultatele sunt preluate din:

```text
pipeline_rebuilt_outputs/reports/model_comparison_rebuilt.csv
```

Figuri utile:

```text
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_new_best_model_v3_rebuilt.png
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_model_2_resnet50_rebuilt.png
pipeline_rebuilt_outputs/plots/model_comparison_rebuilt.png
```

Document Word generat:

```text
Subcapitol_4_2_Evaluarea_Modelelor_Individuale_Test.docx
```

## Date de test folosite

Evaluarea se face pe setul de test:

| Clasa | Numar imagini |
|---|---:|
| benign | 359 |
| malignant | 300 |
| total | 659 |

Nu este nevoie sa detaliezi din nou impartirea train/validation/test, deoarece aceasta a fost explicata in subcapitolul 4.1.

## Rezultate numerice

| Model | Prag | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB0 Rebuilt | 0.40 | 0.8498 | 0.7947 | 0.9033 | 0.8456 | 0.8542 | 289 | 70 | 29 | 271 |
| ResNet50 Rebuilt | 0.60 | 0.8756 | 0.8759 | 0.8467 | 0.8610 | 0.8732 | 323 | 36 | 46 | 254 |

## Interpretare EfficientNetB0

EfficientNetB0 obtine:

- accuracy: `0.8498`;
- recall pentru `malignant`: `0.9033`;
- 271 cazuri maligne clasificate corect;
- 29 fals negative;
- 70 fals pozitive.

Interpretare:

- modelul detecteaza mai multe cazuri maligne;
- are mai putine fals negative decat ResNet50;
- produce mai multe fals pozitive;
- este mai sensibil la clasa `malignant`.

Formulare posibila:

```text
EfficientNetB0 are un comportament mai sensibil pentru clasa malignant, reusind sa identifice 271 dintre cele 300 de cazuri maligne din setul de test. Numarul de fals negative este mai redus, insa acest avantaj este obtinut cu pretul unui numar mai mare de fals pozitive.
```

## Interpretare ResNet50

ResNet50 obtine:

- accuracy: `0.8756`;
- precision: `0.8759`;
- recall pentru `malignant`: `0.8467`;
- 323 cazuri benigne clasificate corect;
- 36 fals pozitive;
- 46 fals negative.

Interpretare:

- modelul are acuratete globala mai buna;
- reduce fals pozitivele;
- este mai precis cand prezice clasa `malignant`;
- rateaza mai multe cazuri maligne decat EfficientNetB0.

Formulare posibila:

```text
ResNet50 obtine cea mai buna acuratete globala si cea mai buna precizie pentru clasa malignant. Modelul este mai strict in atribuirea etichetei pozitive, ceea ce reduce numarul de fals pozitive, dar creste numarul de fals negative.
```

## Comparatie directa

Ideea principala:

- EfficientNetB0 este mai bun la `recall` pentru `malignant`;
- ResNet50 este mai bun la `accuracy`, `precision`, `F1-score` si `balanced accuracy`;
- EfficientNetB0 este mai potrivit daca prioritatea este reducerea cazurilor maligne ratate;
- ResNet50 este mai potrivit daca se doreste un echilibru global mai bun si mai putine alarme false.

Formulare posibila:

```text
Diferenta dintre modele nu indica doar o ierarhie simpla, ci doua comportamente diferite. EfficientNetB0 favorizeaza detectarea leziunilor maligne si reduce fals negativele, in timp ce ResNet50 ofera o performanta globala mai buna si limiteaza fals pozitivele.
```

## Figuri recomandate

Pentru acest subcapitol sunt suficiente 2-3 figuri:

1. Matricea de confuzie EfficientNetB0.
2. Matricea de confuzie ResNet50.
3. Graficul de comparare a modelelor.

Legende posibile:

```text
Figura 4.6. Matricea de confuzie obtinuta pentru EfficientNetB0 pe setul de test.
Figura 4.7. Matricea de confuzie obtinuta pentru ResNet50 pe setul de test.
Figura 4.8. Compararea sintetica a metricilor pentru modelele individuale.
```

## Structura recomandata

1. Introducere scurta: scopul evaluarii individuale.
2. Tabel cu rezultatele modelelor.
3. Interpretarea EfficientNetB0.
4. Interpretarea ResNet50.
5. Comparatie directa intre modele.
6. Concluzie.

## Ce sa nu se repete din 4.1

Evita sa reiei pe larg:

- configurarea `IMG_SIZE`, `BATCH_SIZE`, `SEED`;
- explicatia despre generatorul de test;
- motivatia generala pentru metrici;
- detaliile despre pragurile de cautare;
- organizarea directoarelor de output.

In 4.2 este suficient sa mentionezi doar ca modelele au fost evaluate pe acelasi set de test si ca pragurile folosite sunt cele stabilite anterior.

## Concluzie posibila

```text
Evaluarea individuala pe setul de test arata ca ambele modele invata caracteristici utile pentru clasificarea leziunilor cutanate, dar produc tipuri diferite de erori. EfficientNetB0 este mai sensibil pentru clasa malignant, in timp ce ResNet50 obtine performanta globala mai buna si mai putine fals pozitive. Aceasta analiza justifica folosirea mai multor metrici si pregateste comparatiile ulterioare dintre modele.
```

