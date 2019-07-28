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
python train_gan.py cifar10 ./results/GAN/cifar10_art clustering --clustering_path ./results/clustering/cifar10/ --gpu 0 --seed 1
```