#%% [markdown]
# # Penguin dataset fit.
# Fitting a toy neural network model on a Penguin dataset example.

#%%
import numpy as np
import seaborn as sns
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from toynn.network import Network
from toynn.activations import LINEAR, RELU, SIGMOID, TANH
from toynn.losses import LOGISTIC, SQUARE

#%%
penguins = sns.load_dataset('penguins')
penguins = penguins.dropna(axis=0, how='any')

is_chinstrap = lambda species: \
    'Chinstrap' if species == 'Chinstrap' else 'Other'
penguins['Penguin'] = penguins['species'].apply(is_chinstrap)

X = penguins[['bill_length_mm', 'bill_depth_mm']].values
Y, names = pd.factorize(penguins['Penguin'].values)
Y = np.expand_dims(Y, axis=1)
X.shape, Y.shape

#%%
sns.pairplot(penguins, hue='Penguin')

#%%
sns.scatterplot(data=penguins,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='Penguin')

#%%
nn = Network([2, 5, 5, 1], activation=TANH(),
    outputActivation=TANH())
losses = nn.fit(X, Y, lr=0.005, max_epochs=10000,
    batch_size=len(X))

#%%
import random
from tqdm.notebook import tqdm
from tqdm import tqdm
data = list(zip(X/X.max(), Y/Y.max()))
nn = Network([2, 5, 5, 5, 1], activation=RELU(),
    outputActivation=SIGMOID(), loss=LOGISTIC())

#%%
losses = []
pbar = tqdm(range(1000))
for i in pbar:
    batch = [random.choice(data) for i in range(len(X))]
    loss = nn.fit_batch(batch, 0.001)
    losses.append(loss)
    pbar.set_description(f'loss={loss}')

#%%
sns.lineplot(x=range(len(losses)), y=losses)

#%%
def decision_boundary(X, clf):
    xaxis = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yaxis = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.apply_along_axis(clf, 2, np.dstack([xx, yy]))
    plt.contourf(xx, yy, zz, alpha=0.4)
    print(zz)
decision_boundary(X, lambda x: nn.predict([x]).item())
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y[:, 0])

#%%
# nn.predict(X)
# len(X)
# zz

#%%
nn.forward(X[1])
nn.backward(Y[1])
nn.learn()
nn.get_loss(Y[1])
#%%
len(losses)
# nn.biases

#%% [markdown]
# ## Citations
# - [Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE 9(3):e90081.](https://doi.org/10.1371/journal.pone.0090081)