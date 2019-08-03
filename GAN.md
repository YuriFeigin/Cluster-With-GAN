```
python eval_real_data.py cifar10 is --test_on train
```

```
python eval_real_data.py cifar10 fid
```

```
python eval_real_data.py celeba ndb --img_size 64
```

```
python train_gan.py cifar10 ./results/GAN/cifar10_clu clustering --clustering_path ./results/clustering/cifar10/ --architecture model2 --seed 1
```
```
python train_gan.py cifar100 ./results/GAN/cifar100_clu clustering --clustering_path ./results/clustering/cifar100/ --architecture model2 --seed 1
```
```
python train_gan.py stl10 ./results/GAN/stl10_clu clustering --clustering_path ./results/clustering/stl10/ --architecture model2 --seed 1 --img_size 48
```
```
python train_gan.py celeba ./results/GAN/celeba_clu clustering --clustering_path ./results/clustering/celeba/ --architecture model2 --seed 1 --z_len 512 --n_cluster_hist 3
```


train original CT-GAN CIFAR10 unsupervised
```
python train_gan.py cifar10 ./results/GAN/cifar10_CT_unsup unsup --architecture model2 --seed 1 --train_on train --ACGAN_SCALE 1 --ACGAN_SCALE_G 0.1 --max_iter 100000
```
train original CT-GAN CIFAR10 supervised
```
python train_gan.py cifar10 ./results/GAN/cifar10_CT_sup sup --architecture model2 --seed 1 --train_on train --ACGAN_SCALE 1 --ACGAN_SCALE_G 0.1 --max_iter 100000
```