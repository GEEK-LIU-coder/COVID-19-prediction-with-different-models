import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import pandas as pd


def logistic_model(t, k, p0, r):
    t0 = 19
    tmp = np.exp(r * (t - t0))
    return k * tmp * p0 / (k + (tmp -1)/ p0)


# t = [17, 18, 19, 20, 21, 22]
# t = np.array(t)
# p = [41, 45, 62, 110, 200, 260]
# p = np.array(p)
# popt, pcov = curve_fit(f=logistic_model, xdata=t, ydata=p, bounds=(0, [1000, 100, 100]))
# print('k:capacity   p0:initial_value   r:increase_rate')
# print(popt)
#
# plot1 = plt.plot(t, p, 's', label='confirmed infected people')
# plt.xlabel('time')
# plt.ylabel(('confirm'))
# plt.legend(loc=0)
#plt.show()

# x = np.arange(17, 22, 0.1)
# y=logistic_model(x, 119.666, 50, 37.5277)
# plt.plot(x, y,'p',label='predicted')
# plt.show()



path = "countrydata.csv"
data = pd.read_csv(path)
data = data[data['countryName'] == '中国']
# 获取时间数据
date_list = list(data['dateId'])
# 迭代器
mp = map(lambda x: str(x), date_list)
# 转迭代器为数组
date_list = list(mp)
confirm_list = list(data['confirmedCount'])
print(confirm_list)
#time arr
time_array=list(range(19,len(confirm_list)+19))
#datelist和confirmlist有区别？:无
long_time_array=np.array(range(19,len(date_list)+190))
confirm_arr=np.array(confirm_list)

print('confirm_list:{}'.format(len(confirm_list)))
print('date_list"{}'.format(len(date_list)))

# from scipy.special import expit
# expit(confirm_arr)
#curve fit for past data
popt1,pocv1=curve_fit(logistic_model,time_array,confirm_arr)
print("k:   p0:    r:")
print(popt1)
#用训练好的带popt1参数的函数预测未来150天
p_predict=logistic_model(long_time_array,popt1[0],popt1[1],popt1[2])


#plot for past and predict
plot1=plt.plot(time_array,confirm_arr,'b',label="confirm")
plot2=plt.plot(long_time_array,p_predict,'r',label='predict')
print(p_predict)
plt.title('logistic model')
plt.xlabel('time')
plt.ylabel('confirmed cases')
#一定要加，不加没有标签
plt.legend()
plt.show()



