import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.random import normal, rand
from numpy.random import rand

def main():
    A = np.random.randint(10,size=(2,2))
    B = np.random.randint(10,size=(2,2))

    for row in A:
        print (row)

    for row in B:
        print (row)

    E = A.transpose()
    for row in E:
        print (row)

    c = np.matrix(B)
    ainv = inv(c)

    for row in ainv:
        print (row)


    #
    # t = np.arange(0.0,10.0,0.01)
    # s = np.sin(2*np.pi*t)
    # k = np.cos(2*np.pi*t)
    # plt.plot(t,s)
    # plt.plot(t,k)
    # plt.show()

    x = normal(size=1000)
    plt.hist(x,bins=30)
    plt.show()
    plt.cla()

    # a = rand(1000)
    # b = rand(1000)
    # plt.scatter(a,b)
    # plt.show()
main()
