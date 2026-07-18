# README pentru redactarea subcapitolului 3.2

## Subcapitol vizat

**3.2. Preprocesarea datelor, augmentarea și tratarea dezechilibrului dintre clase**

Acest fișier conține contextul, ideile principale, datele concrete din proiect și fragmentele de cod relevante pentru redactarea subcapitolului 3.2 din lucrarea de licență.

## Contextul lucrării

Lucrarea are ca temă dezvoltarea unei soluții bazate pe inteligență artificială pentru clasificarea imaginilor dermatoscopice ale leziunilor cutanate. Problema finală este formulată ca o clasificare binară între clasele:

- `benign`
- `malignant`

Subcapitolul 3.1 a prezentat setul de date HAM10000 și modul în care imaginile au fost reorganizate în directoare pentru antrenare și testare. Subcapitolul 3.2 continuă partea de implementare și explică modul în care imaginile sunt pregătite înainte de antrenarea modelelor EfficientNetB0 și ResNet50.

## Rolul subcapitolului în lucrare

Subcapitolul 3.2 trebuie să explice:

- de ce este necesară preprocesarea imaginilor înainte de antrenare;
- cum sunt încărcate imaginile din directoarele `train` și `test`;
- cum se face redimensionarea la `224x224` pixeli;
- cum se folosesc funcțiile `preprocess_input` specifice arhitecturilor EfficientNetB0 și ResNet50;
- cum se realizează împărțirea internă train/validation prin `validation_split=0.2`;
- ce transformări de augmentare sunt aplicate imaginilor de antrenare;
- de ce augmentarea este utilă în cazul imaginilor dermatoscopice;
- care este distribuția claselor în proiect;
- cum este tratat dezechilibrul dintre clase prin `compute_class_weight`;
- de ce setul de test nu trebuie augmentat.

Subcapitolul nu trebuie să prezinte rezultatele finale ale modelelor. Acestea aparțin Capitolului 4. De asemenea, nu trebuie să detalieze arhitecturile EfficientNetB0 și ResNet50, deoarece acestea au fost tratate în Capitolul 2 și vor fi reluate în subcapitolul 3.3.

## Fișiere relevante din proiect

Implementarea preprocesării, augmentării și tratării dezechilibrului dintre clase se află în:

```text
pipeline_rebuilt/common.py
pipeline_rebuilt/config.py
```

Structura datasetului reorganizat este:

```text
dataset/new_dataset/
  train/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

## Configurații importante

În `pipeline_rebuilt/config.py` sunt definite valorile comune folosite în pipeline:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
```

Semnificație:

- `IMG_SIZE = (224, 224)` - toate imaginile sunt redimensionate la 224x224 pixeli;
- `BATCH_SIZE = 16` - imaginile sunt procesate în loturi de câte 16;
- `SEED = 42` - seed fix pentru reproductibilitate.

Dimensiunea de `224x224` este potrivită deoarece atât EfficientNetB0, cât și ResNet50 pot lucra cu această dimensiune de intrare.

## Distribuția claselor în proiect

Distribuția reală a imaginilor în datasetul reorganizat este:

| Subset | Benign | Malignant | Total |
|---|---:|---:|---:|
| Train | 1440 | 1197 | 2637 |
| Test | 359 | 300 | 659 |

Observație importantă:

- clasa `benign` este mai numeroasă decât clasa `malignant`;
- dezechilibrul nu este extrem, dar trebuie tratat pentru a evita favorizarea clasei majoritare;
- în context medical, erorile pe clasa `malignant` sunt importante, deoarece pot afecta detectarea cazurilor cu risc.

## Încărcarea imaginilor

În `common.py`, imaginile sunt încărcate cu `ImageDataGenerator` și `flow_from_directory`.

Fragment relevant:

```python
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED,
)
```

Explicații:

- `train_dir` indică directorul `dataset/new_dataset/train`;
- `target_size=IMG_SIZE` redimensionează imaginile la `224x224`;
- `batch_size=BATCH_SIZE` stabilește numărul de imagini procesate simultan;
- `class_mode="binary"` este folosit deoarece problema are două clase;
- `subset="training"` selectează partea de antrenare;
- `shuffle=True` amestecă imaginile pentru antrenare;
- `seed=SEED` păstrează reproductibilitatea.

## Împărțirea train/validation

În proiect, validarea este obținută din setul de antrenare prin:

```python
validation_split=0.2
```

Aceasta înseamnă că 20% din imaginile din `train` sunt folosite pentru validare.

Generatorul de validare este:

```python
val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED,
)
```

Observații:

- pentru validare se folosește `subset="validation"`;
- `shuffle=False` este important pentru evaluare deterministă;
- validarea este folosită în timpul antrenării pentru monitorizarea performanței și pentru callback-uri precum `EarlyStopping`.

## Preprocesarea specifică arhitecturii

Funcția `create_train_val_generators` primește ca parametru funcția de preprocesare:

```python
def create_train_val_generators(train_dir, preprocess_function):
```

Această abordare permite folosirea aceleiași logici pentru ambele arhitecturi:

```python
from tensorflow.keras.applications.efficientnet import preprocess_input
```

pentru EfficientNetB0 și:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input
```

pentru ResNet50.

Rolul funcției `preprocess_input` este să transforme valorile pixelilor în forma așteptată de modelul preantrenat.

## Augmentarea datelor

Augmentarea este aplicată doar pe setul de antrenare, prin `ImageDataGenerator`.

Fragment relevant:

```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_function,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=(0.9, 1.1),
    validation_split=0.2,
)
```

Transformările folosite:

| Transformare | Valoare | Rol |
|---|---:|---|
| `rotation_range` | 20 | permite rotații moderate ale imaginilor |
| `zoom_range` | 0.15 | simulează apropierea sau depărtarea leziunii |
| `horizontal_flip` | True | oglindește imaginile pe orizontală |
| `width_shift_range` | 0.08 | deplasează imaginea pe axa orizontală |
| `height_shift_range` | 0.08 | deplasează imaginea pe axa verticală |
| `brightness_range` | (0.9, 1.1) | modifică ușor luminozitatea |

## De ce este utilă augmentarea

Augmentarea este importantă deoarece:

- crește diversitatea imaginilor văzute de model în timpul antrenării;
- reduce riscul de memorare a imaginilor originale;
- ajută modelul să generalizeze mai bine pe imagini noi;
- simulează variații realiste întâlnite în imagini dermatoscopice;
- contribuie la stabilitatea antrenării în cazul unui set de date limitat.

În cazul imaginilor dermatoscopice, leziunea poate apărea în poziții și orientări diferite, iar luminozitatea poate varia în funcție de condițiile de achiziție. De aceea, rotațiile moderate, deplasările mici, zoom-ul și modificările reduse de luminozitate sunt justificate.

Trebuie menționat că augmentarea este moderată. Nu se folosesc transformări foarte agresive, deoarece acestea ar putea modifica aspectul medical al leziunii.

## Tratarea dezechilibrului dintre clase

Dezechilibrul dintre clase este tratat prin ponderi de clasă calculate automat cu `compute_class_weight`.

Fragment relevant:

```python
classes = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes,
)
class_weights = dict(enumerate(class_weights_array))
```

Aceste ponderi sunt transmise apoi la antrenare:

```python
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights,
)
```

Pe baza distribuției din setul de antrenare:

- benign: 1440 imagini;
- malignant: 1197 imagini.

Ponderile aproximative sunt:

```text
benign:    0.916
malignant: 1.102
```

Interpretare:

- clasa `malignant`, fiind mai puțin reprezentată, primește o pondere mai mare;
- erorile pe clasa `malignant` contribuie mai mult la funcția de pierdere;
- modelul este descurajat să favorizeze clasa majoritară.

Această metodă este preferabilă față de duplicarea simplă a imaginilor, deoarece nu introduce copii identice în setul de antrenare.

## Generatorul pentru evaluare și testare

Pentru evaluare se folosește o variantă fără augmentare:

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

Observații:

- pe test se aplică doar preprocesarea specifică modelului;
- nu se aplică rotații, zoom, flip sau modificări de luminozitate;
- `shuffle=False` păstrează ordinea imaginilor pentru calcularea metricilor, a matricii de confuzie și a rapoartelor de clasificare.

## Idei principale pentru redactare

Textul subcapitolului poate fi structurat astfel:

1. Introducere despre necesitatea preprocesării imaginilor.
2. Descrierea încărcării imaginilor cu `flow_from_directory`.
3. Redimensionarea la `224x224` și folosirea batch size-ului 16.
4. Explicarea preprocesării specifice EfficientNetB0 și ResNet50.
5. Descrierea împărțirii interne train/validation cu `validation_split=0.2`.
6. Prezentarea augmentării și a transformărilor folosite.
7. Justificarea augmentării în contextul imaginilor dermatoscopice.
8. Prezentarea distribuției claselor.
9. Explicarea calculării `class_weight`.
10. Menționarea faptului că testarea se face fără augmentare.
11. Concluzie despre rolul acestei etape în pregătirea antrenării modelelor.

## Figuri recomandate

Pentru acest subcapitol pot fi incluse:

- un grafic cu distribuția claselor în train și test;
- o imagine exemplificativă cu transformări de augmentare;
- eventual o schemă simplă a fluxului: imagine originală -> redimensionare -> preprocesare -> augmentare -> batch de antrenare.

Documentul Word generat pentru acest subcapitol include deja:

```text
Subcapitol_3_2_Preprocesare_Augmentare_Dezechilibru.docx
```

## Formulare de concluzie posibilă

O concluzie potrivită pentru subcapitol:

```text
Prin combinarea redimensionării, preprocesării specifice arhitecturii, augmentării moderate și ponderării claselor, pipeline-ul pregătește datele într-un mod coerent pentru antrenarea modelelor prezentate în subcapitolul următor. Această etapă are un rol esențial deoarece calitatea intrării influențează direct stabilitatea antrenării, capacitatea de generalizare și interpretarea corectă a rezultatelor obținute ulterior.
```

