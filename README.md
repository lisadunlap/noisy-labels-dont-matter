# Stop Caring About Noisy Labels

Contributious:
(1) Show that unless you have a large number of mislabels (>10%) model accuracy doesnt change
(2) Find VI and V2 of datasets (Imagenet, Domain Net. ...)
(3) Explore the tippes of labeling errors
- Segmentation ( boundary, object, etc)
- Classification (egregous,"close",multi-label)
- Class-conditianed querative madels?
- Captioning (TODO)
(4) Ablate diff models/model sizes on different amounts of label noise to see if there is any correlation

Current mislabel methods:
- Random

Current models:
- resnet50

Current Train Datasets:
- Waterbirds
- Imagenette (10 ImageNet classes)
- ImagenetteWoof (10 ImageNet dog classes)
- ImageNet (not tested)

Cureent Robustness Test Datasets:
- Imagenette-C (still need to fix the severity issue)
- Imagenet-A