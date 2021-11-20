import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from scipy.stats import norm

# Đọc dữ liệu từ file vào biến data
data = pd.read_csv(r'H:NhapMonHocSau\HocSau\babies.txt', sep='\s+')
print(data)

# Tạo DataFrame từ dữ liệu và lọc dữ liệu, smoke=1
df_cohutthuoc = pd.DataFrame(data, columns=['bwt', 'smoke']).query('smoke==1')

# Tạo DataFrame từ dữ liệu và lọc dữ liệu, smoke=0
df_khonghutthuoc = pd.DataFrame(data).query('smoke==0')


# nếu muốn vẽ boxplot trên cùng 1 biểu đồ thì dùng như sau, gom nhóm theo cột smoke:
data.boxplot(by='smoke', column=['bwt'], grid=False)
pyplot.show()

tongcohutthuoc = df_cohutthuoc['bwt'].count()
tongkhonghutthuoc = df_khonghutthuoc['bwt'].count()

# lay du lieu mot cot
coHutThuoc = np.array(df_cohutthuoc['bwt'])
khongHutThuoc = np.array(df_khonghutthuoc['bwt'])

print('tong co hut thuoc:'+str(tongcohutthuoc))
print('tong khong hut thuoc:'+str(tongkhonghutthuoc))


print('Các giá trị thống kê trong trường hợp bà mẹ CÓ hút thuốc')
# kurtosis theo pearson
print('Kurtosis la:'+str(kurtosis(coHutThuoc, fisher=False)))
print('Skew la:'+str(skew(coHutThuoc)))
print('Min la:'+str(np.min(coHutThuoc)))
print('Max la:'+str(np.max(coHutThuoc)))
print('TB la:'+str(np.mean(coHutThuoc)))
print('Sdt la:'+str(np.std(coHutThuoc)))
print('var la:'+str(np.var(coHutThuoc)))
print('Median la:'+str(np.median(coHutThuoc)))
print('Q0%:'+str(np.percentile(coHutThuoc, 0)))
print('Q25%:'+str(np.percentile(coHutThuoc, 25)))
print('Q50%:'+str(np.percentile(coHutThuoc, 50)))
print('Q75%:'+str(np.percentile(coHutThuoc, 75)))
q75, q25 = np.percentile(coHutThuoc, [75, 25])
iqr = q75 - q25
print('IQR=Q3-Q1='+str(iqr))
# Cách khác
print("cach khac:")
# theo định nghĩa của Fish thì phân phối chuẩn có kurtosis=0
# Hàm kurtosis của thư viện pandas tính toán Kurtosis của Fisher
# thu được bằng cách lấy Kurtosis của Pearson trừ đi 3.
# Với Kurtosis của Fisher, định nghĩa phân phối chuẩn có kurtosis bằng 0.
# Kurtosis (Fisher)= Kurtosis (Pearson) -3
# để tính kutosis theo pearson thì
# Kurtosis (Pearson)= Kurtosis (Fisher)+3
print('Kurtosis la:', df_cohutthuoc['bwt'].kurtosis()+3)
print('Skew la:', df_cohutthuoc['bwt'].skew())
print('Min la:', df_cohutthuoc['bwt'].min())
print('Max la:', df_cohutthuoc['bwt'].max())
print('TB la:', df_cohutthuoc['bwt'].mean())
print('Sdt la:', df_cohutthuoc['bwt'].std(skipna=True))
print('var la:', df_cohutthuoc['bwt'].var(skipna=True))
print('Median la:', df_cohutthuoc['bwt'].median())
print('Q0%:', df_cohutthuoc['bwt'].quantile(.0))
print('Q25%:', df_cohutthuoc['bwt'].quantile(.25))
print('Q50%:', df_cohutthuoc['bwt'].quantile(.50))
print('Q75%:', df_cohutthuoc['bwt'].quantile(.75))
q75, q25 = np.percentile(coHutThuoc, [75, 25])
iqr = q75 - q25
print('IQR=Q3-Q1=', iqr)


print('Các giá trị thống kê trong trường hợp bà KHÔNG có hút thuốc')
print('Kurtosis la:', kurtosis(khongHutThuoc, fisher=False))
print('Skew la:', skew(khongHutThuoc))
print('Min la:', np.min(khongHutThuoc))
print('Max la:', np.max(khongHutThuoc))
print('TB la:', np.mean(khongHutThuoc))
print('Sdt la:', np.std(khongHutThuoc, ddof=1))
print('var la:', np.var(khongHutThuoc, ddof=1))
print('Median la:', np.median(khongHutThuoc))
print('Q0%:', np.percentile(khongHutThuoc, 0))
print('Q25%:', np.percentile(khongHutThuoc, 25))
print('Q50%:', np.percentile(khongHutThuoc, 50))
print('Q75%:', np.percentile(khongHutThuoc, 75))
q75, q25 = np.percentile(khongHutThuoc, [75, 25])
iqr = q75 - q25
print('IQR=Q3-Q1=', (iqr))


# do thi QQ plot de minh hoa tinh chuan cua du lieu
#stats.probplot(np.array(coHutThuoc), plot=plt)
# stats.show()

qqplot(coHutThuoc, line='s')

#pyplot.xlabel('Cân nặng')
#pyplot.ylabel('Tỉ số Z')
pyplot.title('QQ-plot trong trường hợp bà mẹ có hút thuốc')
pyplot.show()


qqplot(khongHutThuoc, line='s')
#pyplot.xlabel('Cân nặng')
#pyplot.ylabel('Tỉ số Z')
pyplot.title('QQ-plot trong trường hợp bà mẹ không hút thuốc')

pyplot.show()

#qqplot(khongHutThuoc, line='s')
# pyplot.show()


plt.boxplot(coHutThuoc, sym='*')
plt.ylabel('Cân nặng')
plt.title('Boxplot trong trường hợp bà mẹ có hút thuốc')
plt.show()


plt.boxplot(khongHutThuoc, sym='*')
plt.ylabel('Cân nặng')
plt.title('Boxplot trong trường hợp bà mẹ không hút thuốc')
plt.show()

# cách khác
print("ve boxplot theo cach dùng hàm của DataFrame")
df_cohutthuoc.boxplot(column=['bwt'])
plt.show()


plt.hist(coHutThuoc, bins=12)
plt.title('Histogram trong trường hợp bà mẹ có hút thuốc')
plt.xlabel('Cân nặng')
plt.show()

plt.hist(khongHutThuoc, bins=12)
plt.title('Histogram trong trường hợp bà mẹ không hút thuốc')
plt.xlabel('Cân nặng')
plt.show()


# cách khác
print("ve histogam theo cach dùng hàm của DataFrame")
df_cohutthuoc['bwt'].hist(bins=12)
plt.show()


# sns.kdeplot(coHutThuoc)
#sns.kdeplot(coHutThuoc, shade=True, color="r")
#ax = sns.kdeplot(coHutThuoc)

# sns.distplot(coHutThuoc, hist=True, kde=True,
#             bins=12, color = 'darkblue',
#             hist_kws={'edgecolor':'black'},
#             kde_kws={'linewidth': 4})


#stats.probplot(coHutThuoc, plot=plt)

