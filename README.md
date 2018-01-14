# chainer-sklearn-neural-encoder
Code for brain encoding analyses. Uses sklean library for data preprocessing and cross-validation, uses chainer 1.24.0 for training neural networks.

'''
python train.py --input ../data/fasttext_sentences.npy --model ridge --out_file fasttext_ridge
python train.py --input ../data/fasttext_sentences.npy --model mlp --drop 0.05 --n_mid 50 --out_file fasttext_mlp
'''
