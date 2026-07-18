# README pentru redactarea subcapitolului 2.4

## Subcapitol vizat

**2.4. Metrici de evaluare si metode de explicabilitate vizuala**

Acest fisier contine contextul, ideile principale si instructiunile necesare pentru a genera, ulterior, textul academic al subcapitolului 2.4 din lucrarea de licenta.

## Contextul lucrarii

Lucrarea are ca tema dezvoltarea unei solutii bazate pe inteligenta artificiala pentru clasificarea leziunilor cutanate din imagini dermatoscopice. Problema este formulata ca o clasificare binara intre clasele `benign` si `malignant`.

Subcapitolele anterioare au introdus clasificarea imaginilor, retelele neuronale convolutionale, invatarea prin transfer, fine-tuning-ul si arhitecturile EfficientNetB0 si ResNet50. Subcapitolul 2.4 trebuie sa explice cum pot fi evaluate aceste modele si cum pot fi interpretate vizual predictiile lor.

In implementarea proiectului apar urmatoarele elemente relevante:

- predictiile modelelor sunt probabilitati produse de iesirea sigmoid;
- probabilitatile sunt transformate in etichete printr-un prag de decizie;
- sunt folosite metrici precum `accuracy`, `precision`, `recall`, `F1-score`, `balanced accuracy` si `AUC`;
- este generata matricea de confuzie pentru clasele `benign` si `malignant`;
- este trasata curba ROC pentru compararea modelelor;
- sunt folosite metode de explicabilitate vizuala: Grad-CAM, Grad-CAM++ si Score-CAM;
- hartile de activare sunt suprapuse peste imaginea originala pentru a evidentia zonele relevante pentru predictie.

## Rolul subcapitolului in lucrare

Subcapitolul 2.4 inchide Capitolul 2, dupa prezentarea arhitecturilor EfficientNetB0 si ResNet50. Rolul sau este de a introduce instrumentele teoretice prin care modelele vor fi evaluate si interpretate in capitolele urmatoare.

Textul trebuie sa faca legatura intre:

- iesirea probabilistica a modelului;
- alegerea unui prag de clasificare;
- interpretarea corecta a rezultatelor prin metrici;
- importanta evaluarii separate a claselor `benign` si `malignant`;
- necesitatea explicabilitatii vizuale in aplicatiile medicale;
- folosirea metodelor XAI pentru evidentierea regiunilor din imagine care influenteaza predictia.

Subcapitolul trebuie sa pregateasca natural Capitolul 4, unde rezultatele numerice, matricile de confuzie, curbele ROC si imaginile XAI vor fi analizate concret.

## Idei principale care trebuie dezvoltate

1. Evaluarea unui model de clasificare nu trebuie sa se bazeze pe o singura metrica.

2. In probleme medicale, acuratetea poate fi insuficienta, mai ales cand exista dezechilibru intre clase.

3. Matricea de confuzie arata distributia predictiilor corecte si incorecte: true negative, false positive, false negative si true positive.

4. Pentru clasificarea `benign`/`malignant`, erorile de tip false negative sunt deosebit de importante, deoarece o leziune maligna poate fi clasificata gresit ca benigna.

5. Precizia masoara proportia predictiilor pozitive corecte dintre toate predictiile pozitive.

6. Recall-ul masoara proportia exemplelor pozitive identificate corect dintre toate exemplele pozitive reale.

7. F1-score combina precision si recall intr-o singura masura, fiind util cand exista dezechilibru intre clase.

8. Balanced accuracy calculeaza media performantelor pe clase si este utila pentru seturi dezechilibrate.

9. Curba ROC arata relatia dintre rata pozitivelor adevarate si rata falselor pozitive pentru mai multe praguri de decizie.

10. AUC sintetizeaza performanta modelului pe toate pragurile analizate.

11. Pragul de decizie influenteaza eticheta finala si poate fi ajustat in functie de obiectivul aplicatiei.

12. Explicabilitatea vizuala este importanta in domeniul medical deoarece permite inspectarea zonelor din imagine care au influentat predictia modelului.

13. Grad-CAM foloseste gradientii fata de ultimul strat convolutional pentru a construi o harta de activare.

14. Grad-CAM++ este o extensie care poate oferi localizari mai fine in anumite situatii.

15. Score-CAM foloseste scorurile produse de model pentru harti de activare mascate, reducand dependenta directa de gradienti.

16. Metodele XAI nu demonstreaza automat corectitudinea modelului, ci ofera suport pentru interpretarea deciziilor.

17. In lucrare, explicabilitatea vizuala trebuie prezentata ca metoda de analiza si transparentizare, nu ca dovada clinica definitiva.

## Structura recomandata a subcapitolului

Subcapitolul poate fi redactat in 8-10 paragrafe academice, organizate astfel:

1. Introducere: importanta evaluarii riguroase in clasificarea imaginilor medicale.

2. Predictii probabilistice si prag de decizie: explicarea transformarii probabilitatii in eticheta.

3. Matricea de confuzie: definirea TN, FP, FN si TP in contextul `benign`/`malignant`.

4. Metrici clasice: accuracy, precision, recall si F1-score.

5. Balanced accuracy si dezechilibrul dintre clase: de ce este utila pentru date medicale.

6. Curba ROC si AUC: evaluarea performantei pentru mai multe praguri.

7. Necesitatea explicabilitatii vizuale: transparenta si interpretabilitate in aplicatii medicale.

8. Grad-CAM si Grad-CAM++: principiu general si rolul ultimului strat convolutional.

9. Score-CAM: explicarea diferentei fata de metodele bazate pe gradienti.

10. Tranzitie: pregatirea Capitolului 4, unde aceste metrici si metode sunt aplicate pe rezultatele obtinute.

## Continut teoretic recomandat

### Pragul de decizie

Modelele folosite in proiect produc o probabilitate pentru clasa pozitiva prin activarea sigmoid. Aceasta probabilitate trebuie transformata intr-o eticheta finala folosind un prag de decizie. De exemplu, daca probabilitatea pentru clasa `malignant` este mai mare sau egala cu pragul ales, imaginea este clasificata ca `malignant`; altfel, este clasificata ca `benign`.

Este important sa se explice ca pragul nu este doar un detaliu tehnic. Modificarea lui poate schimba echilibrul dintre sensibilitate si numarul de alarme false. In aplicatiile medicale, alegerea pragului trebuie analizata cu atentie.

### Matricea de confuzie

Matricea de confuzie este un instrument central pentru evaluarea clasificarii binare. Ea arata cate exemple au fost clasificate corect si cate au fost clasificate gresit.

Pentru problema lucrarii:

- `TN` reprezinta imaginile benigne clasificate corect ca benigne;
- `FP` reprezinta imaginile benigne clasificate gresit ca maligne;
- `FN` reprezinta imaginile maligne clasificate gresit ca benigne;
- `TP` reprezinta imaginile maligne clasificate corect ca maligne.

In context medical, `FN` trebuie discutat cu atentie, deoarece ratarea unei leziuni maligne poate avea consecinte mai grave decat o alarma falsa.

### Accuracy, precision, recall si F1-score

Accuracy masoara proportia predictiilor corecte din totalul exemplelor evaluate. Desi este intuitiva, aceasta metrica poate fi inselatoare daca setul de date este dezechilibrat.

Precision arata cat de multe dintre predictiile pozitive sunt corecte. Recall arata cat de multe dintre exemplele pozitive reale au fost identificate corect. F1-score combina precision si recall, fiind util atunci cand se doreste un echilibru intre cele doua.

### Balanced accuracy

Balanced accuracy este utila atunci cand clasele nu sunt reprezentate in mod egal. Ea calculeaza media recall-ului pentru fiecare clasa, oferind o imagine mai echilibrata asupra performantei modelului.

In lucrare, aceasta metrica este relevanta deoarece seturile medicale pot contine mai multe exemple dintr-o clasa decat din cealalta. Astfel, un model nu trebuie evaluat doar prin numarul total de predictii corecte, ci si prin comportamentul pe fiecare clasa.

### Curba ROC si AUC

Curba ROC reprezinta grafic relatia dintre rata pozitivelor adevarate si rata falselor pozitive pentru mai multe praguri de clasificare. AUC, aria de sub curba ROC, sintetizeaza capacitatea modelului de a separa cele doua clase.

O valoare AUC mai mare indica, in general, o capacitate mai buna de discriminare intre clase. Totusi, AUC trebuie interpretata impreuna cu celelalte metrici, deoarece nu descrie singura toate tipurile de erori.

### Explicabilitate vizuala

Explicabilitatea vizuala este importanta deoarece modelele CNN sunt adesea considerate dificil de interpretat. In domeniul medical, nu este suficient ca un model sa returneze o predictie; este util sa se observe si care zone ale imaginii au contribuit la decizia modelului.

Metodele Grad-CAM, Grad-CAM++ si Score-CAM genereaza harti de activare care pot fi suprapuse peste imaginea originala. Aceste harti evidentiaza regiunile considerate relevante de model in procesul de clasificare.

### Grad-CAM

Grad-CAM foloseste gradientii scorului de predictie fata de activari din ultimul strat convolutional. Prin combinarea acestor informatii, se obtine o harta care indica zonele importante pentru decizia modelului.

Aceasta metoda este potrivita pentru CNN-uri deoarece ultimele straturi convolutionale pastreaza informatie spatiala despre imagine, dar contin si caracteristici vizuale mai abstracte.

### Grad-CAM++

Grad-CAM++ extinde Grad-CAM si poate produce harti mai detaliate in anumite situatii, mai ales cand exista mai multe regiuni relevante in imagine. In lucrare, aceasta metoda poate fi prezentata ca o alternativa mai rafinata pentru interpretarea vizuala a predictiilor.

### Score-CAM

Score-CAM foloseste hartile de activare pentru a masca imaginea si masoara efectul acestor regiuni asupra scorului produs de model. Spre deosebire de Grad-CAM, Score-CAM nu se bazeaza direct pe gradienti, ci pe modificarea scorului modelului atunci cand anumite zone sunt evidentiate.

## Legatura cu implementarea proiectului

Subcapitolul 2.4 trebuie sa ramana teoretic, dar poate include referinte scurte la implementare:

- modelele sunt evaluate pe setul de test;
- probabilitatile sunt obtinute prin `model.predict`;
- etichetele sunt generate prin compararea probabilitatii cu un prag;
- sunt calculate `accuracy`, `precision`, `recall`, `F1-score` si `balanced accuracy`;
- este generata matricea de confuzie;
- este trasata curba ROC si calculat AUC;
- pentru XAI sunt folosite Grad-CAM, Grad-CAM++ si Score-CAM;
- ultimul strat convolutional este folosit pentru generarea hartilor de activare.

Fragment conceptual care poate fi mentionat:

```text
Evaluarea modelelor combina metrici numerice, precum precision, recall, F1-score si AUC, cu metode de explicabilitate vizuala care evidentiaza zonele imaginii relevante pentru predictie.
```

## Cod care poate fi mentionat in licenta

Subcapitolul 2.4 este teoretic, dar pot fi incluse fragmente scurte pentru a arata cum sunt aplicate metricile si metodele XAI in proiect.

### Transformarea probabilitatilor in etichete

Sursa: `pipeline_rebuilt/compare_models.py`

```python
y_prob = model.predict(gen, verbose=0).ravel()
y_pred = (y_prob >= threshold).astype(int)
```

Explicatie:

Modelul returneaza probabilitati, iar acestea sunt transformate in etichete binare prin compararea cu un prag de decizie. Aceasta etapa este necesara pentru calcularea metricilor de clasificare.

### Metrici de evaluare

Sursa: `pipeline_rebuilt/compare_models.py`

```python
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
balanced_acc = ((tp / (tp + fn + 1e-8)) + (tn / (tn + fp + 1e-8))) / 2
```

Explicatie:

Acest fragment arata calcularea principalelor metrici folosite pentru compararea modelelor. Balanced accuracy este calculata prin media recall-ului pentru clasa pozitiva si clasa negativa.

### Matricea de confuzie

Sursa: `pipeline_rebuilt/evaluate_model.py`

```python
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["benign", "malignant"]
)
```

Explicatie:

Matricea de confuzie permite observarea directa a predictiilor corecte si incorecte pentru cele doua clase. In contextul lucrarii, ea ajuta la interpretarea erorilor pentru `benign` si `malignant`.

### Curba ROC si AUC

Sursa: `pipeline_rebuilt/roc_curve_models.py`

```python
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
```

Explicatie:

Curba ROC analizeaza comportamentul modelului pentru mai multe praguri de decizie, iar AUC sintetizeaza capacitatea modelului de separare a celor doua clase.

### Grad-CAM

Sursa: `pipeline_rebuilt/xai_methods.py`

```python
with tf.GradientTape() as tape:
    conv_outputs, predictions = g_model(input_tensor, training=False)
    score = predictions[:, 0]

grads = tape.gradient(score, conv_outputs)
weights = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
```

Explicatie:

Acest fragment arata principiul Grad-CAM: gradientii sunt folositi pentru a pondera hartile de activare ale ultimului strat convolutional, rezultand o harta care evidentiaza regiunile importante pentru predictie.

### Alegerea ultimului strat convolutional

Sursa: `pipeline_rebuilt/model_registry.py`

```python
last_conv_layer="top_activation"       # EfficientNetB0
last_conv_layer="conv5_block3_out"     # ResNet50
```

Explicatie:

Metodele XAI folosesc ultimul strat convolutional deoarece acesta pastreaza informatia spatiala necesara pentru localizarea zonelor relevante, dar contine si caracteristici vizuale profunde.

### Suprapunerea hartii de activare peste imagine

Sursa: `pipeline_rebuilt/xai_methods.py`

```python
heatmap_colored = plt.get_cmap("jet")(heatmap_resized)[:, :, :3]
return np.clip(0.6 * original_array + 0.4 * heatmap_colored, 0, 1)
```

Explicatie:

Harta de activare este colorata si suprapusa peste imaginea originala, pentru a face mai usor de observat regiunile care au influentat predictia modelului.

## Avantaje care pot fi discutate

### Metrici de evaluare

- permit masurarea obiectiva a performantei;
- evidentiaza tipuri diferite de erori;
- ajuta la compararea EfficientNetB0 si ResNet50;
- sunt utile pentru identificarea comportamentului pe fiecare clasa;
- permit analiza efectului pragului de decizie.

### Explicabilitate vizuala

- ofera indicii despre zonele imaginii care au influentat predictia;
- creste transparenta modelelor CNN;
- poate sprijini analiza erorilor;
- este utila in contextul imaginilor medicale;
- completeaza evaluarea numerica printr-o perspectiva vizuala.

## Limitari care trebuie mentionate

- accuracy poate fi inselatoare pentru seturi dezechilibrate;
- AUC nu descrie singura toate tipurile de erori;
- un prag optim pe un set de date nu garanteaza performanta identica pe alte seturi;
- hartile Grad-CAM, Grad-CAM++ si Score-CAM nu demonstreaza corectitudinea diagnosticului;
- metodele XAI pot evidentia zone aproximative, nu explicatii cauzale definitive;
- interpretarea medicala trebuie realizata cu prudenta si validare clinica.

## Ce trebuie evitat

Evita formulari de tipul:

- "accuracy este suficienta pentru evaluarea modelului";
- "AUC mare inseamna ca modelul este perfect";
- "Grad-CAM explica exact gandirea modelului";
- "harta de activare confirma diagnosticul medical";
- "Score-CAM garanteaza localizarea corecta a leziunii";
- "un model explicabil poate fi folosit clinic fara validare";
- "pragul ales este universal valabil pentru orice set de date".

Evita si intrarea prea detaliata in:

- rezultatele numerice concrete, deoarece acestea apartin Capitolului 4;
- implementarea completa a pipeline-ului, deoarece aceasta apartine Capitolului 3;
- descrierea arhitecturilor EfficientNetB0 si ResNet50, deoarece acestea sunt tratate in 2.3;
- exemple clinice detaliate, daca nu sunt sustinute de date si surse.

## Formulari recomandate

Pot fi folosite formulari de tipul:

- "Evaluarea modelelor de clasificare trebuie realizata printr-un set de metrici complementare."
- "In problemele medicale, analiza erorilor este la fel de importanta ca performanta globala."
- "Matricea de confuzie permite observarea distributiei predictiilor corecte si incorecte."
- "Recall-ul este relevant pentru identificarea corecta a cazurilor pozitive."
- "Balanced accuracy este utila atunci cand clasele sunt dezechilibrate."
- "Curba ROC permite analizarea comportamentului modelului pentru mai multe praguri de decizie."
- "Metodele Grad-CAM, Grad-CAM++ si Score-CAM contribuie la interpretarea vizuala a predictiilor."
- "Hartile de activare trebuie interpretate ca suport explicativ, nu ca dovada clinica definitiva."

## Concepte care pot fi mentionate

- evaluarea modelelor;
- clasificare binara;
- probabilitate;
- prag de decizie;
- matrice de confuzie;
- true negative;
- false positive;
- false negative;
- true positive;
- accuracy;
- precision;
- recall;
- sensitivity;
- specificity;
- F1-score;
- balanced accuracy;
- ROC;
- AUC;
- interpretabilitate;
- explicabilitate vizuala;
- XAI;
- Grad-CAM;
- Grad-CAM++;
- Score-CAM;
- harta de activare;
- ultimul strat convolutional;
- suprapunere heatmap;
- imagini dermatoscopice;
- analiza erorilor.

## Legatura cu restul cuprinsului

Subcapitolul 2.4 trebuie sa continue si sa pregateasca urmatoarele parti ale lucrarii:

- **2.1. Clasificarea imaginilor si retele neuronale convolutionale**: metricile sunt folosite pentru evaluarea clasificarii, iar XAI se aplica pe modele CNN.
- **2.2. Invatarea prin transfer si fine-tuning pentru seturi de date medicale**: modelele adaptate prin transfer learning trebuie evaluate riguros.
- **2.3. Arhitecturile EfficientNetB0 si ResNet50**: metricile si XAI sunt aplicate pentru compararea si interpretarea acestor arhitecturi.
- **4.2. Evaluarea modelelor individuale pe setul de test**: conceptele teoretice despre metrici vor fi folosite pentru analiza rezultatelor.
- **4.3. Compararea rezultatelor, output fusion si analiza erorilor**: matricea de confuzie, F1-score, balanced accuracy si AUC sustin compararea modelelor.
- **4.4. Interpretarea deciziilor prin Grad-CAM, Grad-CAM++ si Score-CAM**: metodele XAI introduse aici vor fi aplicate concret.

## Prompt recomandat pentru ChatGPT

Foloseste urmatorul prompt pentru a genera subcapitolul:

```text
Scrie subcapitolul 2.4 al unei lucrari de licenta, cu titlul "Metrici de evaluare si metode de explicabilitate vizuala".

Contextul lucrarii: lucrarea trateaza dezvoltarea unei solutii bazate pe inteligenta artificiala pentru clasificarea leziunilor cutanate din imagini dermatoscopice. Problema este formulata ca o clasificare binara intre clasele benign si malignant. In subcapitolele anterioare au fost prezentate clasificarea imaginilor, retelele neuronale convolutionale, invatarea prin transfer, fine-tuning-ul si arhitecturile EfficientNetB0 si ResNet50. Acest subcapitol trebuie sa explice metricile folosite pentru evaluarea modelelor si metodele de explicabilitate vizuala folosite pentru interpretarea predictiilor.

Explica rolul pragului de decizie in transformarea probabilitatii produse de sigmoid intr-o eticheta finala. Prezinta matricea de confuzie si termenii true negative, false positive, false negative si true positive in contextul clasificarii benign/malignant. Explica metricile accuracy, precision, recall, F1-score, balanced accuracy, curba ROC si AUC, mentionand de ce nu este suficienta o singura metrica in probleme medicale si de ce dezechilibrul dintre clase trebuie luat in considerare.

Explica apoi necesitatea explicabilitatii vizuale pentru modelele CNN aplicate imaginilor medicale. Prezinta metodele Grad-CAM, Grad-CAM++ si Score-CAM la nivel teoretic, aratand ca acestea genereaza harti de activare care pot fi suprapuse peste imaginea originala pentru a evidentia regiunile relevante pentru predictia modelului. Mentioneaza ca aceste metode cresc transparenta si pot sprijini analiza erorilor, dar nu confirma singure diagnosticul clinic.

Leaga explicatiile de proiect: modelele EfficientNetB0 si ResNet50 sunt evaluate prin accuracy, precision, recall, F1-score, balanced accuracy si AUC; sunt generate matrici de confuzie si curbe ROC; iar interpretarea vizuala foloseste Grad-CAM, Grad-CAM++ si Score-CAM pe ultimul strat convolutional al modelului. Poti include scurte fragmente de cod conceptual, precum transformarea probabilitatilor in etichete prin `y_pred = (y_prob >= threshold).astype(int)`, calcularea metricilor cu sklearn si generarea hartilor Grad-CAM pe baza gradientilor.

Scrie intr-un stil academic, coerent, la persoana a III-a, potrivit pentru o lucrare de licenta. Textul trebuie sa aiba aproximativ 1-2 pagini. Nu prezenta rezultate numerice concrete, deoarece acestea apartin Capitolului 4. Nu transforma subcapitolul intr-o prezentare de cod, ci foloseste codul doar ca suport pentru explicatiile teoretice. Finalul trebuie sa faca o tranzitie naturala catre Capitolul 3 si Capitolul 4, unde implementarea si rezultatele sunt prezentate concret.
```

## Observatii pentru completare ulterioara

Pentru versiunea finala a lucrarii, subcapitolul poate fi completat cu surse bibliografice despre evaluarea clasificarii binare, ROC-AUC, interpretabilitatea modelelor CNN si metode XAI aplicate in imagistica medicala.

Este recomandat ca textul generat sa nu includa rezultate numerice concrete in 2.4. Valorile obtinute pentru EfficientNetB0, ResNet50, matricile de confuzie, curbele ROC si hartile XAI trebuie prezentate in Capitolul 4.
