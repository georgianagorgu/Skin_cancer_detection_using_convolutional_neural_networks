# README pentru redactarea subcapitolului 3.4

## Subcapitol vizat

**3.4. Implementarea predicției, a mecanismului output fusion și a modulelor XAI**

Acest fișier conține contextul, ideile principale, valorile concrete din proiect și fragmentele de cod relevante pentru redactarea subcapitolului 3.4 din lucrarea de licență.

## Contextul lucrării

Lucrarea are ca temă dezvoltarea unei soluții bazate pe inteligență artificială pentru clasificarea imaginilor dermatoscopice ale leziunilor cutanate. Problema finală este formulată ca o clasificare binară între clasele:

- `benign`
- `malignant`

Subcapitolul 3.3 a prezentat antrenarea modelelor EfficientNetB0 și ResNet50. Subcapitolul 3.4 descrie modul în care modelele antrenate sunt folosite după antrenare: predicție pe imagine individuală, output fusion și interpretabilitate vizuală prin metode XAI.

## Rolul subcapitolului în lucrare

Subcapitolul 3.4 trebuie să explice:

- cum este realizată predicția pentru o singură imagine;
- cum este încărcat modelul salvat;
- cum este preprocesată imaginea înainte de predicție;
- cum este transformată probabilitatea în etichetă finală;
- ce praguri de decizie sunt folosite pentru fiecare model;
- cum funcționează mecanismul output fusion;
- cum sunt calculate ponderile modelelor în output fusion;
- ce fișiere sunt generate de etapa de fusion;
- cum este implementat Grad-CAM;
- cum sunt implementate Grad-CAM++, Score-CAM și comparația XAI;
- de ce interpretabilitatea vizuală este importantă într-o aplicație medicală.

Subcapitolul nu trebuie să analizeze în detaliu performanțele finale. Evaluarea completă aparține Capitolului 4.

## Fișiere relevante din proiect

Implementările principale sunt:

```text
pipeline_rebuilt/predict_image.py
pipeline_rebuilt/output_fusion.py
pipeline_rebuilt/grad_cam.py
pipeline_rebuilt/xai_methods.py
pipeline_rebuilt/model_registry.py
pipeline_rebuilt/config.py
```

Artefactele generate sunt:

```text
pipeline_rebuilt_outputs/predictions/
pipeline_rebuilt_outputs/reports/
pipeline_rebuilt_outputs/confusion_matrix/
pipeline_rebuilt_outputs/gradcam/
pipeline_rebuilt_outputs/xai/
```

## Registrul modelelor

Modelele sunt centralizate în `model_registry.py`.

Fragment relevant:

```python
MODEL_SPECS = {
    "efficientnet": ModelSpec(
        key="efficientnet",
        display_name="EfficientNetB0",
        path=EFFICIENTNET_MODEL_PATH,
        preprocess_function=preprocess_eff,
        threshold=EFFICIENTNET_THRESHOLD,
        last_conv_layer="top_activation",
    ),
    "resnet": ModelSpec(
        key="resnet",
        display_name="ResNet50",
        path=RESNET_MODEL_PATH,
        preprocess_function=preprocess_resnet,
        threshold=RESNET_THRESHOLD,
        last_conv_layer="conv5_block3_out",
    ),
}
```

Praguri de decizie:

| Model | Prag | Strat XAI |
|---|---:|---|
| EfficientNetB0 | 0.40 | `top_activation` |
| ResNet50 | 0.60 | `conv5_block3_out` |
| Output fusion | 0.50 | nu se aplică |

Rolul registrului:

- evită duplicarea configurației;
- asociază fiecare model cu funcția corectă de preprocesare;
- păstrează pragul de decizie pentru fiecare arhitectură;
- oferă stratul convoluțional folosit de metodele XAI.

## Predicția pentru o singură imagine

Predicția este implementată în:

```text
pipeline_rebuilt/predict_image.py
```

Argumente:

```python
parser.add_argument("--model", choices=["efficientnet", "resnet"], default="resnet")
parser.add_argument("--image", default=str(DEFAULT_PREDICT_IMAGE))
```

Fluxul predicției:

1. Se alege modelul: `efficientnet` sau `resnet`.
2. Se încarcă specificația modelului din `model_registry.py`.
3. Se încarcă imaginea la dimensiunea `224x224`.
4. Imaginea este convertită în array numeric.
5. Se adaugă dimensiunea de batch.
6. Se aplică funcția de preprocesare specifică modelului.
7. Modelul produce probabilitatea clasei `malignant`.
8. Probabilitatea este comparată cu pragul modelului.
9. Rezultatul este salvat într-un fișier `.txt`.

Fragment relevant:

```python
img = image.load_img(image_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = spec.preprocess_function(img_array)

prob = float(np.squeeze(model.predict(img_array, verbose=0)))
label = "malignant" if prob >= spec.threshold else "benign"
```

## Raportul de predicție

Rezultatul este salvat în:

```text
pipeline_rebuilt_outputs/predictions/
```

Exemplu real generat în proiect:

```text
Imagine: D:\PROIECT_LICENTA\dataset\new_dataset\test\benign\31.jpg
Model: model_2_resnet50_rebuilt.keras
Probabilitate benign: 0.5906
Probabilitate malignant: 0.4094
Prag folosit: 0.60
Predictie finala: benign
```

Interpretare:

- modelul ResNet50 a produs probabilitatea `0.4094` pentru clasa `malignant`;
- pragul ResNet50 este `0.60`;
- deoarece `0.4094 < 0.60`, predicția finală este `benign`.

## Output fusion

Mecanismul output fusion este implementat în:

```text
pipeline_rebuilt/output_fusion.py
```

Scop:

- combină probabilitățile modelelor individuale;
- evită alegerea unui singur model;
- produce o decizie agregată pe baza mai multor arhitecturi.

Output fusion folosește media ponderată a probabilităților.

Fragment relevant:

```python
train_accuracies = np.array(
    [model_train_accuracy(spec) for spec in specs],
    dtype="float32"
)
total = float(np.sum(train_accuracies))
weights = train_accuracies / total
```

Interpretare:

- se calculează acuratețea fiecărui model pe setul de antrenare;
- acuratețile sunt normalizate;
- fiecare model primește o pondere;
- modelul cu acuratețe mai mare contribuie mai mult la decizia finală.

## Combinarea probabilităților

Pentru fiecare model, probabilitatea clasei `malignant` este transformată în două coloane:

```python
two_class_probs = binary_probs_to_two_columns(malignant_prob)
```

Funcția produce:

```text
[P(benign), P(malignant)]
```

Combinarea se face astfel:

```python
weighted_probs += weight * two_class_probs
y_pred = (weighted_probs[:, 1] >= threshold).astype(int)
```

Pragul implicit pentru output fusion este:

```python
OUTPUT_FUSION_THRESHOLD = 0.50
```

## Verificarea ordinii imaginilor

În output fusion, este important ca modelele să prezică imaginile în aceeași ordine.

Fragment relevant:

```python
elif list(reference_gen.filenames) != list(gen.filenames):
    raise ValueError(
        "Generatorii nu au aceeasi ordine a fisierelor. Verifica structura setului de date."
    )
```

Această verificare este necesară deoarece probabilitățile modelelor trebuie combinate pentru aceeași imagine.

## Fișiere generate de output fusion

Output fusion generează:

```text
pipeline_rebuilt_outputs/reports/output_fusion_report.csv
pipeline_rebuilt_outputs/reports/output_fusion_probabilities.csv
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_output_fusion.png
```

Semnificație:

- `output_fusion_report.csv` conține raportul de clasificare;
- `output_fusion_probabilities.csv` conține probabilitățile fiecărui model, ponderile și predicția finală;
- `confusion_matrix_output_fusion.png` conține matricea de confuzie pentru predicția agregată.

## Grad-CAM

Grad-CAM este implementat în:

```text
pipeline_rebuilt/grad_cam.py
```

Scop:

- evidențiază zonele din imagine care au influențat predicția modelului;
- produce o hartă termică suprapusă peste imaginea originală;
- ajută la interpretarea vizuală a deciziei.

Fragment relevant:

```python
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_tensor, training=False)
    class_channel = predictions[:, 0]

grads = tape.gradient(class_channel, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)
```

Explicație:

- se construiește un model intermediar care returnează activările ultimului strat convoluțional și predicția;
- se calculează gradientul scorului pentru clasa `malignant`;
- gradientele sunt mediate pentru a obține ponderile canalelor;
- harta finală indică zonele importante pentru predicție.

## Salvarea vizualizării Grad-CAM

Rezultatul este salvat în:

```text
pipeline_rebuilt_outputs/gradcam/
```

Exemplu:

```text
pipeline_rebuilt_outputs/gradcam/resnet_1_gradcam.png
```

Vizualizarea include:

- imaginea originală;
- heatmap-ul;
- suprapunerea Grad-CAM peste imagine.

## Modulele XAI extinse

Metodele XAI extinse sunt implementate în:

```text
pipeline_rebuilt/xai_methods.py
```

Metode incluse:

- Grad-CAM;
- Grad-CAM++;
- Score-CAM.

Grad-CAM++ rafinează metoda Grad-CAM printr-o ponderare mai detaliată a gradientelor.

Score-CAM folosește hărți de activare ca măști aplicate peste imagine și evaluează contribuția lor prin scorul modelului.

Fragment relevant pentru Score-CAM:

```python
for channel in selected_channels:
    activation_map = activations[:, :, channel]
    resized = tf.image.resize(
        activation_map[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()
    masked_img = img_array * resized[..., np.newaxis]
    score = float(np.squeeze(model.predict(masked_batch, verbose=0)))
```

În proiect, Score-CAM folosește implicit:

```python
max_maps = 32
```

Această limitare reduce costul de calcul.

## Fișiere generate de XAI

Vizualizările XAI sunt salvate în:

```text
pipeline_rebuilt_outputs/xai/
```

Exemplu:

```text
pipeline_rebuilt_outputs/xai/resnet_1_xai_methods.png
```

Figura compară:

- imaginea originală;
- Grad-CAM;
- Grad-CAM++;
- Score-CAM.

## Importanța XAI în context medical

Metodele XAI sunt importante deoarece:

- oferă o explicație vizuală a predicției;
- permit verificarea zonelor pe care modelul le consideră relevante;
- ajută la observarea situațiilor în care modelul se concentrează pe regiuni irelevante;
- cresc transparența sistemului;
- sunt utile în interpretarea rezultatelor din Capitolul 4.

Trebuie menționat că XAI nu înlocuiește evaluarea medicală, ci oferă un suport vizual pentru înțelegerea comportamentului modelului.

## Idei principale pentru redactare

Textul subcapitolului poate fi structurat astfel:

1. Introducere despre utilizarea modelelor după antrenare.
2. Prezentarea registrului de modele.
3. Explicarea predicției pentru o singură imagine.
4. Prezentarea pragurilor de decizie.
5. Explicarea raportului de predicție.
6. Descrierea mecanismului output fusion.
7. Explicarea calculării ponderilor pe baza acurateții pe train.
8. Prezentarea fișierelor generate de output fusion.
9. Descrierea metodei Grad-CAM.
10. Descrierea Grad-CAM++ și Score-CAM.
11. Explicarea importanței XAI în context medical.
12. Concluzie despre rolul acestor module în completarea pipeline-ului.

## Figuri recomandate

Pentru acest subcapitol pot fi incluse:

- o schemă a fluxului de predicție;
- matricea de confuzie pentru output fusion;
- o vizualizare Grad-CAM;
- o comparație între Grad-CAM, Grad-CAM++ și Score-CAM.

Documentul Word generat pentru acest subcapitol este:

```text
Subcapitol_3_4_Predictie_Output_Fusion_XAI.docx
```

## Formulare de concluzie posibilă

```text
În concluzie, componentele descrise în acest subcapitol transformă modelele antrenate în instrumente utilizabile. Predicția individuală oferă o decizie pentru o imagine concretă, output fusion permite combinarea modelelor pentru o decizie agregată, iar modulele XAI oferă suport vizual pentru interpretarea rezultatului. Împreună, aceste module completează pipeline-ul de clasificare și pregătesc baza pentru evaluarea experimentală din capitolul următor.
```

