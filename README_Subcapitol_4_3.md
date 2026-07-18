# README pentru redactarea subcapitolului 4.3

## Subcapitol vizat

**4.3. Compararea rezultatelor, output fusion si analiza erorilor**

Acest fisier contine informatiile esentiale pentru redactarea subcapitolului 4.3. Spre deosebire de 4.2, unde fiecare model este interpretat separat, aici accentul cade pe compararea celor doua modele individuale cu metoda de output fusion si pe analiza erorilor ramase.

## Rolul subcapitolului

Subcapitolul 4.3 trebuie sa explice:

- cum se compara EfficientNetB0, ResNet50 si output fusion;
- ce metrici obtine metoda de fuziune;
- daca output fusion imbunatateste sau nu rezultatele fata de modelele individuale;
- ce compromis apare intre fals pozitive si fals negative;
- cum se comporta metoda combinata atunci cand modelele individuale sunt de acord sau in dezacord;
- ce tipuri de erori raman importante pentru aplicatia medicala.

Subcapitolul nu trebuie sa reia strategia de testare din 4.1 si nici interpretarea separata detaliata a fiecarui model din 4.2.

## Fisiere relevante

Rezultatele modelelor individuale sunt preluate din:

```text
pipeline_rebuilt_outputs/reports/model_comparison_rebuilt.csv
```

Rezultatele output fusion sunt preluate din:

```text
pipeline_rebuilt_outputs/reports/output_fusion_report.csv
pipeline_rebuilt_outputs/reports/output_fusion_probabilities.csv
```

Implementarea output fusion se afla in:

```text
pipeline_rebuilt/output_fusion.py
```

Document Word generat:

```text
Subcapitol_4_3_Compararea_Rezultatelor_Output_Fusion_Analiza_Erorilor.docx
```

Figuri generate special pentru 4.3:

```text
pipeline_rebuilt_outputs/plots/comparison_tradeoff_4_3.png
pipeline_rebuilt_outputs/plots/agreement_analysis_4_3.png
```

## Rezultate comparative

| Metoda | Prag | Accuracy | Precision | Recall | F1-score | Balanced Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientNetB0 | 0.40 | 0.8498 | 0.7947 | 0.9033 | 0.8456 | 0.8542 | 289 | 70 | 29 | 271 |
| ResNet50 | 0.60 | 0.8756 | 0.8759 | 0.8467 | 0.8610 | 0.8732 | 323 | 36 | 46 | 254 |
| Output fusion | 0.50 | 0.8649 | 0.8328 | 0.8800 | 0.8558 | 0.8662 | 306 | 53 | 36 | 264 |

## Ponderile output fusion

Output fusion combina probabilitatile celor doua modele printr-o medie ponderata.

Ponderile calculate pe baza acuratetii pe setul de antrenare sunt:

| Model | Train accuracy | Pondere |
|---|---:|---:|
| EfficientNetB0 | 0.8859 | 0.4927 |
| ResNet50 | 0.9120 | 0.5073 |

Interpretare:

- ponderile sunt apropiate;
- ResNet50 are o influenta usor mai mare;
- EfficientNetB0 influenteaza suficient decizia finala pentru a mentine recall-ul pentru `malignant` peste cel obtinut de ResNet50.

Fragment relevant din `pipeline_rebuilt/output_fusion.py`:

```python
weights = train_accuracies / np.sum(train_accuracies)
weighted_probs += weight * two_class_probs
y_pred = (weighted_probs[:, 1] >= threshold).astype(int)
```

## Interpretarea comparativa

Ideile principale:

- ResNet50 are cea mai buna acuratete globala: `0.8756`;
- ResNet50 are cea mai buna precizie pentru `malignant`: `0.8759`;
- EfficientNetB0 are cel mai bun recall pentru `malignant`: `0.9033`;
- EfficientNetB0 are cele mai putine fals negative: `29`;
- output fusion obtine un compromis intre cele doua modele;
- output fusion are mai putine fals pozitive decat EfficientNetB0: `53` fata de `70`;
- output fusion are mai putine fals negative decat ResNet50: `36` fata de `46`;
- output fusion nu depaseste cel mai bun model individual la toate metricile.

Formulare posibila:

```text
Output fusion nu produce o imbunatatire uniforma fata de modelele individuale, ci o pozitionare intermediara intre comportamentul sensibil al EfficientNetB0 si comportamentul mai strict al ResNet50. Metoda reduce fals pozitivele fata de EfficientNetB0 si reduce fals negativele fata de ResNet50, dar nu obtine nici acuratetea globala maxima, nici recall-ul maxim.
```

## Analiza erorilor

Valorile importante pentru analiza erorilor:

| Aspect analizat | Valoare |
|---|---:|
| Imagini pentru care modelele individuale sunt de acord | 558 |
| Imagini pentru care modelele individuale sunt in dezacord | 101 |
| Output fusion corect cand modelele sunt de acord | 518 |
| Output fusion corect cand modelele sunt in dezacord | 52 |
| Imagini gresite de ambele modele individuale | 40 |
| Imagini corecte doar pentru EfficientNetB0 | 42 |
| Imagini corecte doar pentru ResNet50 | 59 |
| Erori noi produse de fusion cand ambele modele individuale sunt corecte | 0 |

Interpretare:

- cand modelele sunt de acord, decizia este in general stabila;
- cand modelele sunt in dezacord, output fusion are un comportament mai nesigur;
- erorile comune ale modelelor limiteaza performanta metodei combinate;
- metoda de fusion nu introduce erori noi in cazurile in care ambele modele individuale sunt corecte;
- dezacordurile dintre modele sunt zona cea mai importanta pentru imbunatatiri ulterioare.

Formulare posibila:

```text
Analiza dezacordurilor arata ca performanta metodei de output fusion este puternic influentata de cazurile in care cele doua modele produc predictii diferite. Pentru imaginile in care modelele sunt de acord, decizia combinata este stabila, in timp ce pentru cazurile de dezacord rezultatul depinde de diferente relativ mici intre probabilitatile estimate.
```

## Figuri recomandate

Pentru subcapitolul 4.3 sunt recomandate figuri care nu repeta matricile de confuzie deja folosite in 4.1 si 4.2.

Figuri potrivite:

```text
pipeline_rebuilt_outputs/plots/comparison_tradeoff_4_3.png
pipeline_rebuilt_outputs/plots/agreement_analysis_4_3.png
```

Legende posibile:

```text
Figura 4.7. Compromisul dintre fals pozitive si fals negative pentru EfficientNetB0, ResNet50 si output fusion.
Figura 4.8. Corectitudinea output fusion in functie de acordul dintre modelele individuale.
```

## Structura recomandata

1. Introducere scurta despre scopul compararii.
2. Prezentarea metodei de output fusion.
3. Tabel comparativ cu EfficientNetB0, ResNet50 si output fusion.
4. Interpretarea comparativa a metricilor.
5. Analiza compromisului dintre fals pozitive si fals negative.
6. Analiza acordului si dezacordului dintre modele.
7. Concluzie privind utilitatea output fusion.

## Ce sa nu se repete din subcapitolele anterioare

Evita sa reiei pe larg:

- configurarea experimentelor din 4.1;
- explicatia despre `IMG_SIZE`, `BATCH_SIZE`, `SEED`;
- modul de incarcare a generatorului de test;
- explicatiile generale despre metrici;
- analiza separata extinsa a EfficientNetB0 si ResNet50 din 4.2;
- matricile de confuzie deja prezentate anterior;
- curba ROC si graficul general de comparare daca au fost deja incluse in 4.1.

In 4.3 este suficient sa mentionezi valorile modelelor individuale doar pentru a le compara cu output fusion.

## Concluzie posibila

```text
Compararea rezultatelor arata ca output fusion ofera o solutie intermediara intre EfficientNetB0 si ResNet50. Metoda combinata reduce fals pozitivele fata de EfficientNetB0 si reduce fals negativele fata de ResNet50, insa nu depaseste cel mai bun model individual pe toate metricile. Aceasta observatie arata ca fuziunea simpla a probabilitatilor este utila pentru echilibrarea deciziei, dar performanta finala depinde de calitatea si complementaritatea modelelor combinate. Pentru imbunatatiri ulterioare, ar fi necesara optimizarea pragului de decizie sau folosirea unei metode de fuziune adaptate special cazurilor in care modelele sunt in dezacord.
```
