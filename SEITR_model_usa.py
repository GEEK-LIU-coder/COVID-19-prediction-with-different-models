import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

N = 330000000  # 人数
beta = 0.19  # 感染系数
gamma = 0.15  # 恢复率
delta = 0.3  # 治疗系数

Te = 14  # 潜伏期
E_O = 0  # 潜伏初始人数
I_O = 1  # 感染却没住院初始人数

T_O=0    #治疗中
R_O = 0  # 治愈者初始人数

S_O = N-I_O-E_O-R_O-T_O#易感初始人数
T=250#传播时间
INI=[S_O,E_O,I_O,R_O,T_O]
def funcSEITR(inivalue,_):
    #y is the array of changing rate
    y=np.zeros(5)
    x=inivalue
    y[0]=-(beta*x[0]*(x[2]+x[1]))/N
    y[1]=(beta*x[0]*(x[2]+x[1]))/N-x[1]/Te
    y[2]=x[1]/Te-delta*x[2]
    y[3]=gamma*x[4]
    y[4]=delta*x[2]-gamma*x[4]
    return y
T_range=np.arange(0,T+1)
RES=spi.odeint(funcSEITR,INI,T_range)
plt.plot(T_range,RES[:,0],color='darkblue',label='Suspectible')
plt.plot(T_range,RES[:,1],color='orange',label='Exposed')
plt.plot(T_range,RES[:,2],color="red",label='Infection')
plt.plot(T_range,RES[:,3],color='green',label='Recovery')
plt.plot(T_range,RES[:,4],color='blue',label='Under_Treatment')
plt.title('SEITR model')
plt.legend()
plt.xlabel('day')
plt.ylabel('Number')

plt.show()


