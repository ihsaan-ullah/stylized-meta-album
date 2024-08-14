# Domain Adaptation over SMA

This code is adapted from https://github.com/emadeldeen24/AdaTime to support SMA datasets with the addition of SOTA algorithm for Universal Domain Adaptation.

## Requirements:
- Python3
- Pytorch==2.1.1
- Numpy==1.26.0
- scikit-learn==0.19.3
- Pandas==2.1.1
- skorch==0.15.0 (For DEV risk calculations)

## Datasets

Download SMA datasets (https://stylized-meta-album.github.io/) and put them in the folder "data"

## Train Model
To reproduce our results you can execute the command line :
```
python main.py --da_method 'the_method' --dataset 'the_sma_dataset'
```
for example to train CDAN on the 'SPORTS'dataset :

```
python main.py --da_method 'CDAN' --dataset 'SPORTS'
```
To run all our expriments at once you can execute ```uda.bh``` or ```uni_da.bh```

By default, the models test performance will be automatically saved in "SMA_results" at the end of the training.