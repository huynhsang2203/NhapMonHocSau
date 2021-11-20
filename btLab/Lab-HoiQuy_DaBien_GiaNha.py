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


data = pd.read_excel(r'H:\NhapMonHocSau\HocSau\23_HOMES.xls')
#các cột là

df= pd.DataFrame(data,columns=['Selling_Price','List_Price','Area','Acres','Age','Taxes','Rooms','Bedrooms','Baths_full'])
print(df)

result = smf.ols('Selling_Price~List_Price+Area+Acres',df).fit()
print(result.summary())
#nếu dùng 1 biến
result = smf.ols('Selling_Price~List_Price',df).fit()
print(result.summary())

result = smf.ols('Selling_Price~Area',df).fit()
print(result.summary())

result = smf.ols('Selling_Price~Acres',df).fit()
print(result.summary())

#1.Nếu chỉ sử dụng 1 biến x để dự đoán giá nhà, phương trình hồi quy 1 biến dự đoán (predictor) nào sau đây là tốt nhất
#LP
#2.Nếu sử dụng đúng 2 biến dự đoán để dự đoán giá nhà, phương trình hồi quy 2 biến dự đoán (predictor) nào ở trên là tốt nhất? Tại sao?
#LP, LOT hoặc LP, LA
#3.Phương trình hồi quy nào trong số các phương trình hồi quy trên là tốt nhất để dự đoán giá nhà? Tại sao?
#y^=99.2+0.979*LP
#giá trị dự báo là
print(1120+0.972*400000+0.281*3000+465*2)
#ước lượng không tốt


