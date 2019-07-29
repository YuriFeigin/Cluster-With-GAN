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
python train_gan.py celeba ./results/GAN/celeba_clu clustering --clustering_path ./results/clustering/celeba/ --architecture model1 --seed 1 --z_len 512 --DIM_G 64 --DIM_D 64
```