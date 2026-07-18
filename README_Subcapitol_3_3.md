# README pentru redactarea subcapitolului 3.3

## Subcapitol vizat

**3.3. Antrenarea modelelor EfficientNetB0 și ResNet50**

Acest fișier conține contextul, ideile principale, valorile concrete din proiect și fragmentele de cod relevante pentru redactarea subcapitolului 3.3 din lucrarea de licență.

## Contextul lucrării

Lucrarea are ca temă dezvoltarea unei soluții bazate pe inteligență artificială pentru clasificarea imaginilor dermatoscopice ale leziunilor cutanate. Problema finală este formulată ca o clasificare binară între clasele:

- `benign`
- `malignant`

Subcapitolul 3.1 a descris setul de date HAM10000 și organizarea imaginilor. Subcapitolul 3.2 a prezentat preprocesarea, augmentarea și tratarea dezechilibrului dintre clase. Subcapitolul 3.3 descrie antrenarea propriu-zisă a modelelor EfficientNetB0 și ResNet50.

## Rolul subcapitolului în lucrare

Subcapitolul 3.3 trebuie să explice:

- de ce au fost folosite modele preantrenate;
- cum sunt încărcate EfficientNetB0 și ResNet50;
- cum este înlocuit clasificatorul original cu un clasificator binar;
- ce straturi sunt adăugate peste baza convoluțională;
- cum este compilat modelul;
- ce funcție de pierdere și ce metrici sunt folosite;
- cum se desfășoară antrenarea în două etape;
- ce înseamnă etapa de feature extraction;
- ce înseamnă etapa de fine-tuning;
- ce callback-uri sunt folosite;
- unde sunt salvate modelele și graficele de antrenare.

Subcapitolul nu trebuie să prezinte analiza finală a rezultatelor pe setul de test. Aceasta aparține Capitolului 4.

## Fișiere relevante din proiect

Antrenarea modelelor este implementată în:

```text
pipeline_rebuilt/train_efficientnet.py
pipeline_rebuilt/train_resnet.py
pipeline_rebuilt/common.py
pipeline_rebuilt/config.py
```

Rezultatele generate sunt salvate în:

```text
pipeline_rebuilt_outputs/models/
pipeline_rebuilt_outputs/plots/
```

Modelele salvate:

```text
pipeline_rebuilt_outputs/models/new_best_model_v3_rebuilt.keras
pipeline_rebuilt_outputs/models/model_2_resnet50_rebuilt.keras
```

Graficele de antrenare:

```text
pipeline_rebuilt_outputs/plots/training_plot_new_best_model_v3_rebuilt.png
pipeline_rebuilt_outputs/plots/training_plot_model_2_resnet50_rebuilt.png
```

## Configurații comune

În `config.py` sunt definite:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

EFFICIENTNET_NAME = "new_best_model_v3_rebuilt"
RESNET_NAME = "model_2_resnet50_rebuilt"
```

Semnificație:

- imaginile sunt redimensionate la `224x224`;
- loturile de antrenare au `16` imagini;
- seed-ul `42` este folosit pentru reproductibilitate;
- modelele sunt salvate cu nume diferite pentru a putea fi evaluate separat.

## Pregătirea pentru antrenare

Ambele scripturi încep prin:

```python
set_global_seed()
ensure_dirs(MODELS_DIR, PLOTS_DIR)

train_gen, val_gen, class_weights = create_train_val_generators(
    TRAIN_DIR,
    preprocess_input
)
```

Explicații:

- `set_global_seed()` fixează sursele de aleatoriu;
- `ensure_dirs()` creează directoarele de ieșire dacă nu există;
- `create_train_val_generators()` creează generatoarele pentru train și validation;
- `class_weights` sunt folosite pentru tratarea dezechilibrului dintre clase.

## Antrenarea EfficientNetB0

EfficientNetB0 este încărcat preantrenat pe ImageNet:

```python
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
```

Explicații:

- `weights="imagenet"` folosește ponderi preantrenate;
- `include_top=False` elimină clasificatorul original;
- `input_shape=(224, 224, 3)` corespunde imaginilor RGB redimensionate;
- `base_model.trainable = False` îngheață baza în prima etapă.

Clasificatorul adăugat:

```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.45)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.30)(x)
output = layers.Dense(1, activation="sigmoid")(x)
```

Rolul straturilor:

- `GlobalAveragePooling2D` transformă hărțile de caracteristici într-un vector;
- `BatchNormalization` stabilizează distribuția activărilor;
- straturile `Dense` învață combinații relevante pentru clasificarea binară;
- `Dropout` reduce riscul de supraînvățare;
- `Dense(1, activation="sigmoid")` produce probabilitatea clasei pozitive.

## Antrenarea ResNet50

ResNet50 este încărcat similar:

```python
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
```

Clasificatorul adăugat:

```python
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)
```

Diferență față de EfficientNetB0:

- ResNet50 folosește `Dropout(0.5)` după primul strat dens;
- EfficientNetB0 folosește `Dropout(0.45)`.

## Compilarea modelelor

Ambele modele sunt compilate astfel:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="binary_crossentropy",
    metrics=standard_metrics(),
)
```

Funcția `standard_metrics()` este definită în `common.py`:

```python
def standard_metrics():
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
```

Explicații:

- `Adam` este optimizatorul folosit;
- rata de învățare inițială este `2e-4`;
- `binary_crossentropy` este potrivită pentru clasificare binară;
- `accuracy` arată proporția predicțiilor corecte;
- `AUC` măsoară capacitatea modelului de separare a claselor;
- `precision` indică proporția predicțiilor pozitive corecte;
- `recall` indică proporția cazurilor pozitive detectate.

## Etapa 1: feature extraction

În prima etapă, baza preantrenată este înghețată:

```python
base_model.trainable = False
```

Antrenarea:

```python
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    class_weight=class_weights,
)
```

Caracteristici:

- maximum `10` epoci;
- se antrenează doar clasificatorul adăugat;
- baza preantrenată rămâne neschimbată;
- se folosesc ponderile de clasă.

Rol:

- clasificatorul nou învață să folosească reprezentările extrase de modelul preantrenat;
- se reduce riscul de modificare agresivă a ponderilor ImageNet.

## Etapa 2: fine-tuning

În a doua etapă, baza devine parțial antrenabilă.

Pentru EfficientNetB0:

```python
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False
```

Pentru ResNet50:

```python
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
```

Recompilarea modelului:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=8e-6),
    loss="binary_crossentropy",
    metrics=standard_metrics(),
)
```

Antrenarea:

```python
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=callbacks,
    class_weight=class_weights,
)
```

Caracteristici:

- maximum `8` epoci;
- rata de învățare scade la `8e-6`;
- EfficientNetB0 deblochează ultimele `40` de straturi;
- ResNet50 deblochează ultimele `30` de straturi;
- restul straturilor rămân înghețate.

Rol:

- modelul se adaptează mai fin la imaginile dermatoscopice;
- straturile finale învață caracteristici mai specifice domeniului medical;
- rata mică de învățare reduce riscul de degradare a ponderilor preantrenate.

## Callback-uri folosite

Callback-urile sunt aceleași pentru ambele modele:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
    ),
]
```

Explicații:

- `EarlyStopping` oprește antrenarea dacă `val_auc` nu se mai îmbunătățește;
- `restore_best_weights=True` restaurează cele mai bune ponderi;
- `ReduceLROnPlateau` reduce rata de învățare când `val_loss` stagnează;
- `ModelCheckpoint` salvează doar modelul cu cea mai bună valoare `val_auc`;
- `patience=4` pentru EarlyStopping;
- `patience=2` pentru ReduceLROnPlateau;
- `min_lr=1e-6` stabilește limita minimă a ratei de învățare.

## Salvarea modelelor și a graficelor

EfficientNetB0:

```python
plot_history(history1, history2, EFFICIENTNET_PLOT_PATH)
```

Fișiere generate:

```text
pipeline_rebuilt_outputs/models/new_best_model_v3_rebuilt.keras
pipeline_rebuilt_outputs/plots/training_plot_new_best_model_v3_rebuilt.png
```

ResNet50:

```python
plot_history(history1, history2, RESNET_PLOT_PATH, title_suffix=" - Rebuilt")
```

Fișiere generate:

```text
pipeline_rebuilt_outputs/models/model_2_resnet50_rebuilt.keras
pipeline_rebuilt_outputs/plots/training_plot_model_2_resnet50_rebuilt.png
```

Graficele urmăresc:

- `Train Accuracy`;
- `Validation Accuracy`;
- `Train Recall`;
- `Validation Recall`;
- `Train Loss`;
- `Validation Loss`;
- `Train AUC`;
- `Validation AUC`.

## Idei principale pentru redactare

Textul subcapitolului poate fi structurat astfel:

1. Introducere despre rolul antrenării modelelor.
2. Justificarea folosirii transfer learning.
3. Prezentarea pregătirii comune pentru ambele modele.
4. Descrierea EfficientNetB0 și a clasificatorului adăugat.
5. Descrierea ResNet50 și a clasificatorului adăugat.
6. Explicarea compilării modelelor.
7. Prezentarea metricilor folosite.
8. Descrierea etapei 1: feature extraction.
9. Descrierea etapei 2: fine-tuning.
10. Explicarea callback-urilor.
11. Menționarea salvării modelelor și graficelor.
12. Concluzie despre comparabilitatea celor două antrenări.

## Figuri recomandate

Pentru acest subcapitol pot fi incluse:

- o schemă a fluxului de antrenare;
- graficul de antrenare pentru EfficientNetB0;
- graficul de antrenare pentru ResNet50.

Documentul Word generat pentru acest subcapitol este:

```text
Subcapitol_3_3_Antrenarea_Modelelor_EfficientNetB0_ResNet50.docx
```

## Formulare de concluzie posibilă

```text
În concluzie, antrenarea modelelor EfficientNetB0 și ResNet50 este construită ca un proces controlat, reproductibil și comparabil. Ambele modele folosesc același set de date preprocesat, aceeași strategie de împărțire train/validation, aceleași metrici și aceeași logică de antrenare în două etape. Diferențele dintre rezultate pot fi analizate ulterior în funcție de particularitățile arhitecturale ale celor două rețele și de modul în care fiecare extrage caracteristici relevante din imaginile dermatoscopice.
```

