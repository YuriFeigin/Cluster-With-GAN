# Reproduce clustering results (paper Cluster with GANs)
this document will guide you how to reproduce all the clustering results from the paper cluster with GANs  
## CIFAR10
to train the network run
```
python train_clustering.py cifar10 ./results/clustering/cifar10 --architecture model2 --seed 1
```
to see the results on tensorboard run
```
tensorboard --logdir ./results/clustering/ --samples_per_plugin "scalars=10000,images=50" --port 6066
```
loss curves (tensorboard):
<img src="images/clustering/cifar10/loss.png"/>
reconstruct images (tensorboard):
<img src="images/clustering/cifar10/rec.png"/>

(optional) in parallel to the training the following script can be run to calculate the clustering (run on cpu) the script will save results to tensorboard
```
python cluster_analysis.py ./results/clustering/cifar10 full
```
clustering curve (tensorboard):
<img src="images/clustering/cifar10/cluster_curve.png"/>

after the training finish you can create a image of samples from each cluster by running
```
python cluster_analysis.py ./results/clustering/cifar10 draw
```
<img src="images/clustering/cifar10/cluster_samples.png"/>
and calculate the final results of ACC, NMI and ARI by running

```
python cluster_analysis.py ./results/clustering/cifar10 final
```
## CIFAR100
to train the network on cifar100 run
```
python train_clustering.py cifar100 ./results/clustering/cifar100 --architecture model2 --seed 1
```
## STL10
to train the network on stl10 run
```
python train_clustering.py stl10 ./results/clustering/stl10 --architecture model2 --seed 1 --img_size 32
```
## celeba
to train the network on celeba run
```
python train_clustering.py celeba ./results/clustering/celeba --architecture ALI_orig_celeba --seed 1 --z_len 512 --max_iter 200000
```


 ```
tensorboard --logdir ./cache/tsne
```

