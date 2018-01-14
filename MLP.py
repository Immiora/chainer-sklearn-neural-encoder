import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt

def split_list(l, wanted_parts=1):
    length = len(l)
    return [ l[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

def shuffle_list(l):
    ind = np.random.permutation(range(len(l)))
    return [l[i] for i in ind]

def make_batches(x, t, batch_size, shuffle=False):
    def make_batch_indices(L, batch_size):
        s = int(np.round(len(L) / float(batch_size)))
        return split_list(L, s)
    batch_indices = make_batch_indices(L=shuffle_list(range(x.shape[0])) if shuffle else range(x.shape[0]),
                                                                                                batch_size=batch_size)
    xb, tb = [], []
    [xb.append(x[i, :]) for i in batch_indices]
    [tb.append(t[i, :]) for i in batch_indices]
    return [xb, tb]

class _MLP(chainer.Chain):
    def __init__(self, n_input, n_mid, n_output):
        super(_MLP, self).__init__(
            l1=L.Linear(n_input, n_mid),
            b1=L.BatchNormalization(n_mid),
            l2=L.Linear(n_mid, n_mid),
            b2=L.BatchNormalization(n_mid),
            l3=L.Linear(n_mid, n_output),
        )
        
    def forward(self, x, train=True, drop=0.):
        self.h1 = F.dropout(self.b1(F.elu(self.l1(x))), drop, train)
        self.h2 = F.dropout(self.b2(F.elu(self.l2(self.h1))), drop, train)
        self.y = self.l3(self.h2)
        return self

    def __call__(self, x, t, train=True, drop=0.):
        self.forward(x, train, drop)
        self.loss = F.mean_squared_error(t, self.y)
        return self

class MLP:
    def __init__(self, n_input, n_output, n_mid=10, lr=1e-03, w_decay=0,
                 batch_size=1, shuffle_batches=True, drop=0., n_epochs=10, toplot = False):
        self.model = _MLP(n_input, n_mid, n_output)
        self.optimizer = chainer.optimizers.Adam(alpha=lr)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate=w_decay))
        self.params = type('test', (object,), {})()
        self.params.w_decay = w_decay
        self.params.drop = drop
        self.params.batch_size = batch_size
        self.params.n_epochs = n_epochs
        self.shuffle_batches = shuffle_batches
        self.no_improve_lim = 50
        self.no_improve = 0
        self.min_val_loss = 1e+06
        self.toplot = toplot

    def fit(self, xtr, ttr, xv=None, tv=None):
        if self.toplot: plt.ion()
        if self.toplot: plt.figure()

        for ie in range(self.params.n_epochs):
            x, t = make_batches(xtr, ttr, self.params.batch_size, shuffle=self.shuffle_batches)
            self.iepoch = ie
            self.tloss, self.tacc, self.vloss, self.vacc = [], [], [], []
            [self.tloss.append([]) for i in x]
            [self.tacc.append([]) for i in x]

            for ix, ib in enumerate(range(len(x))):
                self.ibatch = ix
                self.model(x[ib], t[ib], train=True, drop=self.params.drop)
                loss = self.model.loss
                y_hat_train = self.model.y.data
                self.model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()
                self.tloss[ib].append(loss.data)
                self.tacc[ib].append(np.array([np.corrcoef(y_hat_train[:,i], t[ib][:,i])[0,1] for i in range(t[ib].shape[-1])]))
                self.print_report('train')

            if (xv is not None) & (tv is not None):
                model_copy = self.model.copy()
                model_copy(xv, tv, train=False, drop=0.)
                vloss = model_copy.loss.data
                vy_hat = model_copy.y.data
                self.vloss.append(vloss)
                self.vacc.append(np.array([np.corrcoef(vy_hat[:,i], tv[:,i])[0,1] for i in range(tv.shape[-1])]))
                self.print_report('val')
            print('---------')
            if self.toplot: self.plot_fit(t=t[-1], y_hat=y_hat_train)
        if self.toplot: plt.close()
        if self.toplot: plt.ioff()

    def predict(self, xts):
        model_copy = self.model.copy()
        model_copy.forward(xts, train=False, drop=0.)
        return model_copy.y.data

    def plot_fit(self, t, y_hat):
        plt.clf()
        plt.plot(t[:500 if t.shape[0]>500 else t.shape[0], 1362], label='true')
        plt.plot(y_hat[:500 if t.shape[0]>500 else t.shape[0], 1362], label='predicted')
        plt.legend()
        plt.title('Loss: ' + str(self.tloss[-1][-1]))
        plt.pause(0.5)

    def print_report(self, report='train'):
        if report == 'train':
            print("Epoch: " + str(self.iepoch + 1) + " , batch: " + str(self.ibatch + 1) +
                  ", train loss: " + str(round(self.tloss[self.ibatch][-1], 4)) +
                  ", max train acc: " + str(round(max(self.tacc[self.ibatch][-1]), 4)))

        elif report == 'val':
            print("\t\t\t val loss: " + str(round(self.vloss[-1], 4)) +
                  ", max val acc: " + str(round(max(self.vacc[-1]), 4)))

            print("\t\t\t min. val loss: " + str(round(self.min_val_loss, 4)) +
                                        " (" + str(self.no_improve) + "/" + str(self.no_improve_lim) + ")")
        elif report == 'test':
            print("Test loss: " + str(round(self.test_loss, 4)) +
                  ", max test acc: " + str(max(self.test_acc)))
