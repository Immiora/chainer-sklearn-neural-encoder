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

def trim(x, n):
    n_trim = x.shape[0] - x.shape[0]/n * n
    return x[:-n_trim] if n_trim > 0 else x

def roll(x, n_roll=0):
    return np.roll(x, shift=n_roll, axis=0)

def reshape3(x, dim2):
    return x.reshape(x.shape[0]/dim2, dim2, x.shape[-1])

def make_batches(ktrain, batch_size, shuffle=False):
    def make_batch_indices(L, batch_size):
        s = int(np.round(len(L) / float(batch_size)))
        batch_size1 = len(L) / s + 1
        return split_list(L, batch_size1)

    batch_indices = make_batch_indices(L=range(ktrain[0].shape[0]), batch_size=batch_size)
    n_batches = max([len(i) for i in batch_indices])

    kbtrain = []
    for i_batch in range(n_batches):
        b = [i[i_batch] for i in batch_indices if i_batch < len(i)]
        if shuffle: b = shuffle_list(b)
        kbtrain.append(((np.array(ktrain[0][[i for i in b], :]),
                         np.array(ktrain[1][[i for i in b], :]))))

    return [[i[0] for i in kbtrain], [i[1] for i in kbtrain]]

class _LSTM(chainer.Chain):
    def __init__(self, n_input, n_mid, n_output):
        super(_LSTM, self).__init__(
            l1=L.LSTM(n_input, n_mid),
            l2=L.LSTM(n_mid, n_mid),
            l3=L.Linear(n_mid, n_output),
        )

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def forward(self, x, train=True, drop=0.):
        self.h1 = F.dropout(self.l1(x), drop, train)
        self.h2 = F.dropout(self.l2(self.h1), drop, train)
        self.y = self.l3(self.h2)
        return self

    def __call__(self, x, t, train=True, drop=0.):
        self.forward(x, train, drop)
        self.loss = F.mean_squared_error(t, self.y)
        return self

class LSTM:
    def __init__(self, n_input, n_output, n_mid=10, lr=1e-03, w_decay=0, n_back=1, batch_size=1, drop=0., n_epochs=10, toplot = False):
        self.model = _LSTM(n_input, n_mid, n_output)
        self.optimizer = chainer.optimizers.Adam(alpha=lr)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(rate=w_decay))
        self.params = type('test', (object,), {})()
        self.params.n_back = n_back
        self.params.w_decay = w_decay
        self.params.drop = drop
        self.params.batch_size = batch_size
        self.params.n_epochs = n_epochs
        self.no_improve_lim = 50
        self.no_improve = 0
        self.min_val_loss = 1e+06
        self.toplot = toplot

    def fit(self, xtr, ttr, xv=None, tv=None):
        if (xv is not None) & (tv is not None): xv, tv = trim(xv, self.params.n_back), trim(tv, self.params.n_back)
        if (xv is not None) & (tv is not None): xv, tv = reshape3(xv, self.params.n_back), reshape3(tv, self.params.n_back)
        if self.toplot: plt.ion()
        if self.toplot: plt.figure()

        for ie in range(self.params.n_epochs):
            x, t = roll(xtr), roll(ttr)
            x, t = trim(x, self.params.n_back), trim(t, self.params.n_back)
            x, t = reshape3(x, self.params.n_back), reshape3(t, self.params.n_back)
            x, t = make_batches([x, t], self.params.batch_size, shuffle=False)

            self.iepoch = ie
            self.tloss, self.tacc, self.vloss, self.vacc = [], [], [], []
            [self.tloss.append([]) for i in x]
            [self.tacc.append([]) for i in x]
            if hasattr(self.model, 'reset_state'): self.model.reset_state()

            for ix, ib in enumerate(range(len(x))):
                self.ibatch = ix
                loss, y_hat_train = self.forward_over_time(x[ib], t[ib], train=True, drop=self.params.drop)
                self.model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                self.optimizer.update()
                self.tloss[ib].append(loss.data / float(self.params.n_back))
                self.tacc[ib].append(np.array([np.corrcoef(y_hat_train.reshape(-1, y_hat_train.shape[-1])[:,i],
                                                         t[ib].reshape(-1, t[ib].shape[-1])[:, i])[0,1] for i in range(t[ib].shape[-1])]))
                self.print_report('train')

            if (xv is not None) & (tv is not None):
                vloss, vy_hat = self.forward_over_time(xv, tv, train=False, drop=0.)
                self.vloss.append(vloss / float(self.params.n_back))
                self.vacc.append(np.array([np.corrcoef(vy_hat.reshape(-1, vy_hat.shape[-1])[:,i],
                                        tv.reshape(-1, tv.shape[-1])[:, i])[0,1] for i in range(tv.shape[-1])]))
                self.print_report('val')
            print('---------')
            if self.toplot: self.plot_fit(t=t[-1].reshape(-1, t[-1].shape[-1]), y_hat=y_hat_train.reshape(-1, y_hat_train.shape[-1]))
        if self.toplot: plt.close()
        if self.toplot: plt.ioff()


    def forward_over_time(self, x, t, train, drop):
        if train:
            loss, y_hat = 0, []
            for i in range(x.shape[1]):
                self.model(x[:, i, :], t[:, i, :], train=True, drop=drop)
                loss += self.model.loss
                y_hat.append(self.model.y.data)
            y_hat = np.swapaxes(np.array(y_hat), 0, 1)
            return loss, y_hat
        else:
            model_copy = self.model.copy()
            if hasattr(model_copy, 'reset_state'): model_copy.reset_state()
            loss, y_hat = 0, []
            for i in range(x.shape[1]):
                model_copy(x[:, i, :], t[:, i, :], train=False, drop=0.)
                loss += model_copy.loss.data
                y_hat.append(model_copy.y.data)
            y_hat = np.swapaxes(np.array(y_hat), 0, 1)
            return loss, y_hat

    def predict(self, xts):
        x = trim(xts, self.params.n_back)
        r = xts[x.shape[0]:, :]
        x = reshape3(x, self.params.n_back)
        model_copy = self.model.copy()
        if hasattr(model_copy, 'reset_state'): model_copy.reset_state()
        y_hat = []
        for i in range(x.shape[1]):
            model_copy.forward(x[:, i, :], train=False, drop=0.)
            y_hat.append(model_copy.y.data)
        y_hat = np.swapaxes(np.array(y_hat), 0, 1)
        y_hat = y_hat.reshape(-1, y_hat.shape[-1])
        for i in range(r.shape[0]):
            model_copy.forward(np.expand_dims(r[i, :], axis=0), train=False, drop=0.)
            y_hat = np.append(y_hat, model_copy.y.data, axis=0)
        return y_hat

    def plot_fit(self, t, y_hat):
        plt.clf()
        t = t[:500 if t.shape[0]>500 else t.shape[0], 1362]
        y_hat = y_hat[:500 if y_hat.shape[0]>500 else y_hat.shape[0], 1362]
        t = (t-np.mean(t))/np.std(t)
        y_hat = (y_hat - np.mean(y_hat)) / np.std(y_hat)
        plt.plot(t, label='true')
        plt.plot(y_hat, label='predicted')
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
