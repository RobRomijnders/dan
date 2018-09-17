import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
from collections import OrderedDict
style.use('fivethirtyeight')

lambda_values = [0.001, 0.003, 0.005, 0.008, 0.01]
miou_values = [35.6, 36.9, 37.5, 36.4, 36.4]

plt.bar(lambda_values, miou_values, 0.001)
plt.xlabel('\lambda_c')
plt.ylabel('mIOU on Cityscapes val')
plt.ylim([35., 38.])
plt.show()