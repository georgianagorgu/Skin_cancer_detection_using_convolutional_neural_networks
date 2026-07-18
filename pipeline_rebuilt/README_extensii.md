# Extensii pentru validare externa, output fusion si XAI

## Set extern

Pentru validare externa, pune imaginile intr-un director cu structura:

```text
dataset/external_test/
  benign/
  malignant/
```

Ruleaza:

```powershell
python pipeline_rebuilt/external_validation.py
```

Pentru alt director:

```powershell
python pipeline_rebuilt/external_validation.py --data-dir D:\cale\catre\set_extern
```

## Output fusion

Scriptul salveaza probabilitatile individuale pentru clasele `benign` si `malignant`, calculeaza ponderile din acuratetea pe Train si aplica media ponderata:

```powershell
python pipeline_rebuilt/output_fusion.py
```

Rezultate:

```text
pipeline_rebuilt_outputs/reports/output_fusion_probabilities.csv
pipeline_rebuilt_outputs/reports/output_fusion_report.csv
pipeline_rebuilt_outputs/confusion_matrix/confusion_matrix_output_fusion.png
```

## Exemple corecte si incorecte

```powershell
python pipeline_rebuilt/classification_examples.py --model resnet
```

Rezultate:

```text
pipeline_rebuilt_outputs/classification_examples/resnet/correct/
pipeline_rebuilt_outputs/classification_examples/resnet/incorrect/
pipeline_rebuilt_outputs/classification_examples/resnet/resnet_classification_examples.csv
pipeline_rebuilt_outputs/classification_examples/resnet/resnet_classification_examples.png
```

## XAI

Grad-CAM, Grad-CAM++ si Score-CAM:

```powershell
python pipeline_rebuilt/xai_methods.py --model resnet --image dataset/new_dataset/test/benign/1.jpg
```

Rezultat:

```text
pipeline_rebuilt_outputs/xai/
```
