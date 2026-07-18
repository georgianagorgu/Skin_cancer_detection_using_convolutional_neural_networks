# README pentru redactarea subcapitolului 3.1

## Subcapitol vizat

**3.1. Descrierea setului de date HAM10000 si organizarea imaginilor**

Acest fisier contine contextul, ideile principale, datele concrete din proiect si instructiunile necesare pentru generarea textului academic al subcapitolului 3.1 din lucrarea de licenta.

## Contextul lucrarii

Lucrarea are ca tema dezvoltarea unei solutii bazate pe inteligenta artificiala pentru clasificarea imaginilor dermatoscopice ale leziunilor cutanate. Problema finala este formulata ca o clasificare binara intre clasele `benign` si `malignant`.

Capitolul 2 a prezentat fundamentele teoretice: clasificarea imaginilor, retelele neuronale convolutionale, transfer learning, arhitecturile EfficientNetB0 si ResNet50, metricile de evaluare si metodele XAI. Capitolul 3 trece de la teorie la implementarea concreta a solutiei propuse.

Subcapitolul 3.1 trebuie sa descrie setul de date utilizat si modul in care imaginile au fost organizate pentru antrenarea si evaluarea modelelor.

## Rolul subcapitolului in lucrare

Subcapitolul 3.1 deschide Capitolul 3 si stabileste baza practica a implementarii. El trebuie sa explice:

- ce este setul de date HAM10000;
- ce fisiere si directoare exista in proiect;
- ce informatii contine fisierul `HAM10000_metadata.csv`;
- cum se trece de la diagnosticele originale multiclasă la problema binara `benign` / `malignant`;
- cum este organizat datasetul final in directoare de antrenare si testare;
- de ce aceasta structura este potrivita pentru incarcarea automata a imaginilor cu `flow_from_directory`;
- cum se conecteaza organizarea datelor cu antrenarea modelelor din subcapitolele urmatoare.

Subcapitolul nu trebuie sa prezinte rezultate numerice de performanta ale modelelor. Acestea apartin Capitolului 4. De asemenea, nu trebuie sa detalieze complet arhitecturile EfficientNetB0 si ResNet50, deoarece acestea au fost tratate in Capitolul 2 si vor fi reluate in subcapitolele de implementare.

## Setul de date HAM10000

In proiect exista urmatoarele fisiere si directoare relevante:

```text
dataset/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  hmnist_8_8_L.csv
  hmnist_8_8_RGB.csv
  hmnist_28_28_L.csv
  hmnist_28_28_RGB.csv
  new_dataset/
```

Pentru implementarea proiectului sunt importante in special:

- `HAM10000_metadata.csv` - fisierul cu metadate si etichete diagnostice;
- `HAM10000_images_part_1/` si `HAM10000_images_part_2/` - directoarele cu imaginile dermatoscopice originale;
- `dataset/new_dataset/` - datasetul reorganizat pentru clasificare binara.

Fisierul `HAM10000_metadata.csv` contine coloanele:

```text
lesion_id,image_id,dx,dx_type,age,sex,localization
```

Semnificatia principala a acestor campuri:

- `lesion_id` - identificatorul leziunii;
- `image_id` - identificatorul imaginii, folosit pentru asocierea metadatelor cu fisierul imagine;
- `dx` - diagnosticul / clasa originala;
- `dx_type` - tipul confirmarii diagnosticului;
- `age` - varsta pacientului;
- `sex` - sexul pacientului;
- `localization` - localizarea anatomica a leziunii.

In subcapitol trebuie pus accent in special pe `image_id` si `dx`, deoarece acestea sunt esentiale pentru organizarea imaginilor pe clase.

## Diagnosticele originale din HAM10000

Setul HAM10000 contine sapte categorii diagnostice originale. In proiect, distributia din `HAM10000_metadata.csv` este:

| Cod diagnostic | Numar imagini | Observatie |
|---|---:|---|
| `akiec` | 327 | Keratoze actinice / carcinom intraepitelial |
| `bcc` | 514 | Carcinom bazocelular |
| `bkl` | 1099 | Leziuni benigne de tip keratoza |
| `df` | 115 | Dermatofibrom |
| `mel` | 1113 | Melanom |
| `nv` | 6705 | Nevi melanocitari |
| `vasc` | 142 | Leziuni vasculare |

Aceasta distributie trebuie folosita pentru a explica faptul ca setul original este dezechilibrat. Clasa `nv` are mult mai multe exemple decat celelalte clase, iar clase precum `df` sau `vasc` au un numar redus de imagini.

## Formularea problemei binare

Lucrarea nu foloseste clasificarea in sapte clase, ci transforma problema intr-o clasificare binara:

```text
benign
malignant
```

Aceasta alegere este motivata de obiectivul lucrarii: diferentierea intre leziuni benigne si leziuni maligne in imagini dermatoscopice.

In text trebuie mentionat ca iesirea modelului este o singura probabilitate, interpretata ca probabilitatea apartenentei la clasa `malignant`. Ulterior, aceasta probabilitate este comparata cu un prag de decizie pentru a obtine eticheta finala.

Nu este necesar sa fie discutata clinic in detaliu fiecare clasa originala. Este suficient sa se explice ca diagnosticele originale au fost reorganizate intr-o structura binara compatibila cu modelele implementate.

## Organizarea datasetului in proiect

Datasetul final folosit de modelele implementate este organizat astfel:

```text
dataset/new_dataset/
  train/
    benign/
    malignant/
  test/
    benign/
    malignant/
```

Aceasta structura este importanta deoarece permite incarcarea automata a imaginilor si etichetelor cu `ImageDataGenerator.flow_from_directory`.

Distributia imaginilor in datasetul reorganizat este:

| Subset | Clasa | Numar imagini |
|---|---|---:|
| `train` | `benign` | 1440 |
| `train` | `malignant` | 1197 |
| `test` | `benign` | 359 |
| `test` | `malignant` | 300 |

Total:

- train: 2637 imagini;
- test: 659 imagini;
- total dataset reorganizat: 3296 imagini.

Impartirea pastreaza aproximativ raportul 80% pentru antrenare si 20% pentru testare.

## Elemente de implementare relevante

### Configurarea cailor si parametrilor

Sursa: `pipeline_rebuilt/config.py`

```python
DATASET_DIR = BASE_DIR / "dataset" / "new_dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
```

Acest fragment trebuie folosit pentru a arata ca proiectul centralizeaza caile si parametrii principali. Dimensiunea `IMG_SIZE = (224, 224)` este folosita pentru redimensionarea imaginilor inainte de introducerea lor in retelele EfficientNetB0 si ResNet50.

### Incarcarea imaginilor cu flow_from_directory

Sursa: `pipeline_rebuilt/common.py`

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

Acest fragment trebuie explicat academic:

- `target_size=IMG_SIZE` redimensioneaza imaginile la 224x224 pixeli;
- `batch_size=BATCH_SIZE` stabileste numarul de imagini procesate intr-un lot;
- `class_mode="binary"` indica faptul ca problema are doua clase;
- `subset="training"` si `subset="validation"` separa intern datele de antrenare si validare;
- `shuffle=True` este folosit la antrenare pentru amestecarea imaginilor;
- `shuffle=False` este folosit la validare pentru a pastra ordinea predictiilor.

### Augmentarea datelor

Sursa: `pipeline_rebuilt/common.py`

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

Augmentarea trebuie prezentata ca o metoda de crestere a variabilitatii datelor de antrenare. Transformarile aplicate nu schimba eticheta imaginii, dar ajuta modelul sa devina mai robust la variatii de pozitie, orientare si luminozitate.

Nu trebuie insistat excesiv pe augmentare in 3.1; detaliile pot fi dezvoltate in subcapitolul despre antrenarea modelelor.

### Ponderarea claselor

Sursa: `pipeline_rebuilt/common.py`

```python
classes = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes,
)
class_weights = dict(enumerate(class_weights_array))
```

Acest fragment poate fi mentionat pentru a explica faptul ca dezechilibrul dintre clase este tratat si la nivel de antrenare. Ponderile de clasa cresc importanta exemplelor din clasa mai putin reprezentata si reduc tendinta modelului de a favoriza clasa majoritara.

## Imagini care pot fi incluse in subcapitol

Pentru ilustrarea datasetului pot fi folosite imagini din:

```text
dataset/new_dataset/train/benign/
dataset/new_dataset/train/malignant/
```

Exemple concrete gasite in proiect:

```text
dataset/new_dataset/train/benign/100.jpg
dataset/new_dataset/train/malignant/10.jpg
```

O figura utila pentru subcapitol:

**Figura 3.1. Exemple de imagini dermatoscopice din structura binara a proiectului**

Figura poate avea doua imagini alaturate:

- o imagine din clasa `benign`;
- o imagine din clasa `malignant`.

Scopul figurii este doar ilustrativ. Nu trebuie interpretata clinic in detaliu.

## Structura recomandata a subcapitolului

Subcapitolul poate fi redactat in aproximativ 2-3 pagini, organizat astfel:

1. Introducere: trecerea de la partea teoretica la organizarea concreta a datelor.
2. Prezentarea setului HAM10000 si a fisierului `HAM10000_metadata.csv`.
3. Explicarea diagnosticelor originale si a dezechilibrului dintre clase.
4. Justificarea transformarii problemei intr-o clasificare binara `benign` / `malignant`.
5. Descrierea structurii `dataset/new_dataset/train/test`.
6. Prezentarea distributiei imaginilor in subseturile de antrenare si testare.
7. Explicarea incarcarii automate a imaginilor prin `flow_from_directory`.
8. Mentionarea redimensionarii la 224x224, a batch size-ului si a validarii interne.
9. Mentionarea scurta a augmentarii si ponderarii claselor.
10. Concluzie: organizarea datelor permite antrenarea si evaluarea modelelor din subcapitolele urmatoare.

## Ton si stil recomandat

Textul trebuie scris:

- in stil academic;
- la persoana a III-a;
- coerent si explicativ;
- potrivit pentru o lucrare de licenta;
- fara exprimari colocviale;
- fara liste excesive in forma finala, cu exceptia tabelelor sau figurilor;
- cu accent pe legatura dintre dataset si implementarea modelelor.

Termenii `benign`, `malignant`, `train`, `test`, `HAM10000_metadata.csv`, `flow_from_directory`, `ImageDataGenerator` pot fi pastrati in engleza, deoarece sunt termeni tehnici sau denumiri din proiect.

## Ce sa nu fie inclus in subcapitol

Evita:

- rezultate de acuratete, AUC, precizie, recall sau F1-score;
- compararea EfficientNetB0 vs ResNet50;
- matrici de confuzie sau curbe ROC;
- explicatii detaliate despre Grad-CAM, Grad-CAM++ sau Score-CAM;
- descrierea completa a codului de antrenare;
- concluzii clinice definitive;
- afirmatii ca sistemul poate inlocui medicul dermatolog.

Aceste aspecte apartin subcapitolelor urmatoare sau Capitolului 4.

## Legatura cu subcapitolele urmatoare

Finalul subcapitolului trebuie sa faca tranzitia catre urmatoarele parti ale Capitolului 3:

- pregatirea datelor permite antrenarea modelelor de transfer learning;
- structura binara permite folosirea iesirii sigmoid si a functiei `binary_crossentropy`;
- separarea train/test permite evaluarea obiectiva a modelelor;
- organizarea pe directoare permite integrarea directa cu pipeline-ul implementat.

O formulare posibila pentru final:

```text
Prin urmare, organizarea setului HAM10000 intr-o structura binara, separata in subseturi de antrenare si testare, reprezinta etapa de baza a implementarii. Aceasta permite incarcarea automata a imaginilor, aplicarea preprocesarii necesare si pregatirea datelor pentru antrenarea modelelor EfficientNetB0 si ResNet50, prezentata in subcapitolele urmatoare.
```

## Prompt recomandat pentru generarea subcapitolului

```text
Scrie subcapitolul 3.1, intitulat "Descrierea setului de date HAM10000 si organizarea imaginilor", pentru o lucrare de licenta despre clasificarea binara a leziunilor cutanate in imagini dermatoscopice folosind modele de deep learning.

Textul trebuie sa fie academic, coerent, la persoana a III-a, potrivit pentru o lucrare de licenta, de aproximativ 2-3 pagini.

Include:
- descrierea setului HAM10000;
- explicarea fisierului HAM10000_metadata.csv si a campurilor importante;
- distributia diagnosticelor originale: akiec 327, bcc 514, bkl 1099, df 115, mel 1113, nv 6705, vasc 142;
- explicarea transformarii problemei intr-o clasificare binara benign/malignant;
- structura datasetului final: dataset/new_dataset/train/benign, train/malignant, test/benign, test/malignant;
- distributia imaginilor reorganizate: train benign 1440, train malignant 1197, test benign 359, test malignant 300;
- explicarea incarcarii datelor cu ImageDataGenerator si flow_from_directory;
- mentionarea parametrilor IMG_SIZE=(224,224), BATCH_SIZE=16, SEED=42;
- explicarea scurta a augmentarii si a ponderarii claselor.

Poti include doua tabele:
1. distributia diagnosticelor originale din HAM10000;
2. distributia imaginilor in datasetul binar reorganizat.

Poti include si fragmente scurte de cod din config.py si common.py, dar codul trebuie folosit doar ca suport pentru explicatie, nu ca prezentare completa de implementare.

Nu include rezultate de performanta, matrici de confuzie, curbe ROC sau metode XAI, deoarece acestea apartin capitolelor urmatoare.
```
