from sys import path

path.append('../')
import pandas as pd
from tl.util import read_data, plt_encoding_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train, test_A, _ = read_data()
plt_encoding_error()

corr = train.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# sns.pairplot(train)
plt.show()

# from string import ascii_letters
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
#
# sns.set(style="white")
#
# # Generate a large random dataset
# rs = np.random.RandomState(33)
# d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                  columns=list(ascii_letters[26:]))
#
# # Compute the correlation matrix


