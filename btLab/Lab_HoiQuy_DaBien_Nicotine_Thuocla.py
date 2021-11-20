import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from scipy.stats.stats import pearsonr
import statsmodels.formula.api as smf
#import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot


from patsy import dmatrices, dmatrix, demo_data
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import norm
from sklearn.model_selection import train_test_split


data = pd.read_excel(r'H:\NhapMonHocSau\HocSau\04_CIGARET.xls')
#các cột là

df= pd.DataFrame(data,columns=['KgTar',	'KgNic','KgCO','MnTar','MnNic','MnCO','FLTar','FLNic','FLCO'])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = df['KgTar']
x2 = df['KgCO']
x3 = df['KgNic']

ax.scatter3D(x1, x2, x3, c=x3, cmap='Greens')
ax.set_xlabel('Kg nhựa')
ax.set_ylabel('Kg CO')
ax.set_zlabel('Kg Nicotin')
#ax.xlabel("height")
plt.title('Lượng nicotine trong thuốc lá theo lượng nhựa và CO')
plt.show()

result = smf.ols('KgNic~KgTar+KgCO',df).fit()

print(result.summary())

#Kết quả là: Nicotine=1.59+0.0231*Tar-0.0525*CO
#Kết quả ở summary cho thấy các giá trị P-value>alpha nên kết quả này không có y nghĩa thống kê
#R-squared:9.9%
#Adj. R-squared:1.7%
#giá trị này quá thấp
#nên phương trình hồi quy trên không thể sử dụng để dự đoán lượng nicotine trong thuốc lá
#khi biết lượng nhựa và CO trong thuốc lá



