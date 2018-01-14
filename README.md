# chainer-sklearn-neural-encoder
Code for brain encoding analyses. Uses sklean library for data preprocessing and cross-validation, uses chainer 1.24.0 for training neural networks.

Input: x.file
Output: y.file

Model to learn the mapping is a parameter: ridge / mlp / lstm.

```
python train.py --input ../data/stimuli.npy --output ../data/brain_recs.npy --model ridge --out_file stim_ridge
python train.py --input ../data/stimuli.npy --output ../data/brain_recs.npy --model mlp --drop 0.05 --n_mid 50 --out_file stim_mlp
python train.py --input ../data/stimuli.npy --output ../data/brain_recs.npy --model lstm --drop 0.1 --n_mid 50 --n_back 10 --out_file stim_lstm
```
