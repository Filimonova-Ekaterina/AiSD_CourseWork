import math
import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import sys
import timeit

def LinearSearch(arr, element):
    for i in range (len(arr)):
        if arr[i] == element:
            return i
    return "Элемент не найден"

def BinarySearchIterative(arr, element):
    left = 0
    right = len(arr)-1
    index = -1
    while (left <= right) and (index == -1):
        mid = (left+right)//2
        if arr[mid] == element:
            index = mid
        else:
            if element<arr[mid]:
                right = mid -1
            else:
                left = mid +1
    if (index==-1): return "Элемент не найден"
    else: return index

def BinarySearchRecursive(arr, element, left, right):
    if left > right:
        return "Элемент не найден"
    mid = (left + right) // 2 
    if arr[mid] == element:
        return mid 
    elif arr[mid] < element:
        return BinarySearchRecursive(arr, element, mid + 1, right)
    else:
        return BinarySearchRecursive(arr, element, left, mid - 1) 

def JumpSearch (arr, element):
    length = len(arr)
    jump = int(math.sqrt(length))
    left, right = 0, 0
    while left < length and arr[left] <= element:
        right = min(length - 1, left + jump)
        if arr[left] <= element and arr[right] >= element:
            break
        left += jump;
    if left >= length or arr[left] > element:
        return "Элемент не найден"
    right = min(length - 1, right)
    i = left
    while i <= right and arr[i] <= element:
        if arr[i] == element:
            return i
        i += 1
    return "Элемент не найден"

def FibonacciSearch(arr, element):
    el_min_2 = 0
    el_min_1 = 1
    el = el_min_1 + el_min_2
    while (el < len(arr)):
        el_min_2 = el_min_1
        el_min_1 = el
        el = el_min_1 + el_min_2
    index = -1;
    while (el > 1):
        i = min(index + el_min_2, (len(arr)-1))
        if (arr[i] < element):
            el = el_min_1
            el_min_1 = el_min_2
            el_min_2 = el - el_min_1
            index = i
        elif (arr[i] > element):
            el = el_min_2
            el_min_1 = el_min_1 - el_min_2
            el_min_2 = el - el_min_1
        else :
            return i
    if(el_min_1 and index < (len(arr)-1) and arr[index+1] == element):
        return index+1;
    return "Элемент не найден"

def ExponentialSearch(arr, element):
    if arr[0] == element:
        return 0
    index = 1
    while index < len(arr) and arr[index] <= element:
        index = index * 2
    return BinarySearchIterative( arr[:min(index, len(arr))], element)

def InterpolationSearch(arr, element):
    low = 0
    high = (len(arr) - 1)
    while low <= high and element >= arr[low] and element <= arr[high]:
        index = low + int(((float(high - low) / ( arr[high] - arr[low])) * ( element - arr[low])))
        if arr[index] == element:
            return index
        if arr[index] < element:
            low = index + 1;
        else:
            high = index - 1;
    return "Элемент не найден"



random.seed(1)
size=[]
n=1000000
first=[]
last=[]
mid=[]
rand=[]
dop=[]
for number in range(500000, n + 1, 100000):
    arr = [j for j in range(number)]
    element = arr[randint(0, number-1)]
    elapsed_time = timeit.timeit(lambda: BinarySearchIterative(arr, element), number=number) / number
    rand.append(elapsed_time)
    elapsed_time = timeit.timeit(lambda: BinarySearchRecursive(arr, element,0,number-1), number=number) / number
    first.append(elapsed_time)
    elapsed_time = timeit.timeit(lambda: ExponentialSearch(arr, element), number=number) / number
    last.append(elapsed_time)
    elapsed_time = timeit.timeit(lambda: FibonacciSearch(arr, element), number=number) / number
    mid.append(elapsed_time)
    elapsed_time = timeit.timeit(lambda: InterpolationSearch(arr, element), number=number) / number
    dop.append(elapsed_time)
    size.append(number)

#print(size)
#print(first)
#print(mid)
#print(last)
#print(rand)
plt.plot(size,first, label='Binary recursive search',marker='o',markersize = 4,color="k")
plt.plot(size,rand, label='Binary iterative search',marker='o',markersize = 4,color="r")
plt.plot(size,last, label='Exponential search',marker='o',markersize = 4,color="b")
plt.plot(size,mid, label='Fibonacci search',marker='o',markersize = 4,color="y")
plt.plot(size,dop, label='Interpolation search',marker='o',markersize = 4,color="g")
#x=np.array(size)
#y=np.array(rand)
#log_x=np.log(x)
#b,a=np.polyfit(log_x,y,1)
#print(a, ' + log(x)*',b)
#poly=a+b*log_x
#plt.scatter(x,y,color="blue")
#plt.plot(x,poly,color='red')
plt.xlabel('Size of array')
plt.ylabel('Time to search')
plt.legend()
plt.savefig("comp3.png")
plt.show()


#Теоретическая часть

#def t(n):
#    if n == 1:
#        return 1
#    if n>1:
#        return t(n//2)+1
#size_teor = []
#time_teor1=[]
#time_teor2=[]
#time_teor3=[]
#time_teor4=[]
#time_teor5=[]
#time_teor6=[]
#n = 10000
#for i in range(10):
#    size_teor.append(n)
#    time_teor1.append(4*math.log(n,2))
#    time_teor2.append(t(n//2)+1)
#    time_teor3.append(2* math.sqrt(n))
#    time_teor4.append(3*math.log(n,3)+math.log(n,1.6))
#    time_teor5.append(5*math.log(n,2))
#    time_teor6.append(math.log(math.log(n),2))
#    n+=2000
#print(time_teor1)
#plt.plot(size_teor,time_teor1, label='Binary Iterative',marker='o',markersize = 4,color="r")
#plt.plot(size_teor,time_teor2, label='Binary Recursive',marker='o',markersize = 4,color="y")
#plt.plot(size_teor,time_teor3, label='Jump',marker='o',markersize = 4,color="m")
#plt.plot(size_teor,time_teor4, label='Fibonacci',marker='o',markersize = 4,color="0.8")
#plt.plot(size_teor,time_teor5, label='Exponential',marker='o',markersize = 4,color="b")
#plt.plot(size_teor,time_teor6, label='Interpolation',marker='o',markersize = 4,color="g")
