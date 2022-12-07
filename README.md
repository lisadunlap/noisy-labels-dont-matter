# Stop Caring About Noisy Labels

## Contributious:

1. Show that unless you have a large number of mislabels (>10%) model accuracy doesnt change

2. Find VI and V2 of datasets (Imagenet, Domain Net. ...)

3. Explore the types of labeling errors
- Segmentation ( boundary, object, etc)
- Classification (egregous,"close",multi-label)
- Class-conditianed querative madels?
- Captioning (TODO)

4. Ablate diff models/model sizes on different amounts of label noise to see if there is any correlation

Current mislabel methods:
- Random

Current models:
- resnet50

Current Train Datasets:
- Waterbirds
- Imagenette (10 ImageNet classes)
- ImagenetteWoof (10 ImageNet dog classes)
- ImageNet (not tested)

Current Robustness Test Datasets:
- Imagenette-C (still need to fix the severity issue)
- Imagenet-A

## TODOs
- Compare training-from scratch to finetuning (need to find non-imagenet dataset) []
- Get results for other ImagenetC corruptions []
- Compare clean val to not clean val [X]

### What types of mislabels are there?

Given a matrix showing the probability of each class being mislabled given a class to mislabel
- random (assumes that each class is going to mislabel as another class with equal probability)
- eggeregous (mislabled class is mislabled to least likeley class)
- expected (sample mislabled class according to said matrix)

Dataset mislabels vs natural mislabels
- datasets contain samples from a ton of different settings (more similar to adding in OOD samples) while many use-cases contain mislabels from a roughly similar environment 

## The Important Part (commands)

To train a model on clean data:
```
python main.py --config configs/{DATASET}/base.yaml 
```

**Note** the way we wave and load checkpoints is based on `exp.run` so if you have the same `exp.run` you will be overwriting any checkpoints that already exist for than run name

To train a model on 5% random noise
```
python main.py --config configs/{DATASET}/random.yaml noise.p=0.05
```
could also use base.yaml and change `noise.method=random` and `exp.run`