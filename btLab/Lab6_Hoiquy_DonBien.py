import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import pearsonr
#import statsmodels.formula.api as smf
#import statsmodels.api as sm
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
#from patsy import dmatrices, dmatrix, demo_data
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.gofplots import ProbPlot
from matplotlib import pyplot
from scipy.stats import norm


df = pd.read_csv(r'H:\NhapMonHocSau\HocSau\crabs.txt', sep='\s+')
print(df)

# vẽ scatter plot
# scatter plot showing actual data
plt.plot(df['postsz'], df['presz'], 'o')
plt.xlabel('Postmost size')
plt.ylabel('Premolt size')
plt.title('Postmolt vs premolt')

plt.show()

# tính hệ số tương quan
print('He so tuong quan la:')
print(pearsonr(df['postsz'], df['presz']))


# truyền Y trước, X sau.
result = smf.ols('presz~postsz', df).fit()
# print(result.params)
print(result.summary())
# print(result.conf_int())


sales_pred = result.predict()

# Plot regression against actual data
# plt.figure(figsize=(12, 6))
# scatter plot showing actual data
plt.plot(df['postsz'], df['presz'], 'o')
plt.plot(df['postsz'], sales_pred, 'r', linewidth=2)   # regression line
plt.xlabel('Postmolt size')
plt.ylabel('Premolt size')
plt.title('Postmolt vs premolt')

plt.show()


# vẽ 4 đồ thị để kiểm định lại phương trình hồi quy
# Đồ thị 1
model_fitted_y = result.fittedvalues
# model residuals
model_residuals = result.resid
# normalized residuals
model_norm_residuals = result.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = result.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = result.get_influence().cooks_distance[0]
# dffits's distance, from statsmodels internals
# or use dffits's distance
model_dffits = result.get_influence().summary_frame()["dffits"]

# đồ thị vẽ giá trị fitted và phần dư
plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(model_fitted_y, model_residuals, data=df,
                                  lowess=True,
                                  scatter_kws={'alpha': 0.5},
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')

plt.show()

"""
qqplot(model_residuals, line='s')
#pyplot.xlabel('Cân nặng')
#pyplot.ylabel('Tỉ số Z')
pyplot.title('QQ-plot trong trường hợp bà mẹ có hút thuốc')
pyplot.show()
"""

#QQ = ProbPlot(model_norm_residuals)
QQ = ProbPlot(model_norm_residuals)
#QQ = ProbPlot(model_residuals)
plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
plot_lm_2.axes[0].set_title('Normal Q-Q')
plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
plot_lm_2.axes[0].set_ylabel('Standardized Residuals')
# annotations
abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]
for r, i in enumerate(abs_norm_resid_top_3):
    plot_lm_2.axes[0].annotate(i,
                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                   model_norm_residuals[i]))

plt.show()

plot_lm_3 = plt.figure()
plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_3.axes[0].set_title('Scale-Location')
plot_lm_3.axes[0].set_xlabel('Fitted values')
plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

# annotations
abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
for i in abs_norm_resid_top_3:
    plot_lm_3.axes[0].annotate(
        i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]))

plt.show()

plot_lm_4 = plt.figure()
plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
sns.regplot(model_leverage, model_norm_residuals,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
plot_lm_4.axes[0].set_ylim(-3, 5)
plot_lm_4.axes[0].set_title('Residuals vs Leverage')
plot_lm_4.axes[0].set_xlabel('Leverage')
plot_lm_4.axes[0].set_ylabel('Standardized Residuals')
plt.show()


# annotations
leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
for i in leverage_top_3:
    plot_lm_4.axes[0].annotate(i,
                               xy=(model_leverage[i],
                                   model_norm_residuals[i]))
plt.show()


# tinh khoang sai so cua he so beta0, beta1 khong dung thu vien ols
# dung cong thuc trong slide bai giang
PostMolt = df.postsz[:-1]
x = PostMolt

print('Tong so dong:', result.nobs)


xichma_binh = (result.resid*result.resid).sum()/(result.nobs-2)
xichma = np.sqrt(xichma_binh)
x_gach = df.postsz[:-1].mean()
x_gach_binhphuong = x_gach*x_gach
x_x_ganh = ((x-x_gach)*(x-x_gach)).sum()


se_beta0 = xichma*np.sqrt((1/result.nobs) + x_gach_binhphuong/x_x_ganh)

se_beta1 = xichma/np.sqrt(x_x_ganh)

print('Se_beta0:', se_beta0)
print('Se_beta1:', se_beta1)


t = (-1)*stats.t.ppf(0.025, result.nobs-2, 0, 1)

print('t là:', t)
beta0 = result.params[0]
beta1 = result.params[1]

print('beta0 là:', beta0)

print(beta0 - t * se_beta0, beta0 + t * se_beta0)

print('beta1 là:', beta1)

print(beta1 - t * se_beta1, beta1 + t * se_beta1)


# print(result.params[1])

PostMolt = df.postsz[:-1]
# tính khoảng dự báo cho y
# trong trường hợp đã loại bỏ các điểm influence ảnh hưởng đến phương trình hồi quy
# thì cần xây dựng lại phương trình hồi quy ứng với trường hợp đã loại bỏ
# giả sử phương trình hồi quy này đã được kiểm định và xây dựng khi đã loại bỏ
# thì mới dùng phương trình này để dự báo cho y
# vào dự báo theo công thức sau:
#PreMolt= df.premolt[:-1]
# xichma binh phuong = tong e binh phuong/n-2
# se(y0)=xichma*canbac2(1+1/n+(x0-xgach)binhphuong/tong(xi-x_gach)binhphuong)
# y_mu=y0+-t(n-2,alphachia2)*se(y0)


xichma_binh = (result.resid*result.resid).sum()/(result.nobs-2)
xichma = np.sqrt(xichma_binh)
PostMolt = df.postsz[:-1]
x = PostMolt
x0 = 85
x_gach = df.postsz[:-1].mean()
x_x_ganh = ((x-x_gach)*(x-x_gach)).sum()
x0_x_gach = ((x0-x_gach)*(x0-x_gach))
se_y0 = xichma*np.sqrt(1+1/result.nobs+x0_x_gach/x_x_ganh)
y0 = result.params[0]+result.params[1]*x0
print('y0 là:', y0)

t = (-1)*stats.t.ppf(0.025, result.nobs-2, 0, 1)

print(y0 - t * se_y0, y0 + t * se_y0)
