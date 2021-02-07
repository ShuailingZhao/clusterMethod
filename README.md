## 聚类方法dbscan

#### testDbScan 聚类团装点
最小密度：半径内小于最小密度就认为是噪声，会中断一个聚类的样本，一个类变成两个类，或者理解为聚类点连续性的最小容忍度
input:  最小密度，半径，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testDbScan -testData=../data/benchmark_hepta.dat
```

#### testDbScanImage 聚类车道
最小密度：半径内小于最小密度就认为是噪声，会中断一个聚类的样本，一个类变成两个类，或者理解为聚类点连续性的最小容忍度
input:  最小密度，半径，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testDbScanImage -testImg=../data/39-1001150009181229150644600.png
```

#### testGMMImage 聚类车道不合适
该算法只适合团装样本点的聚类，每个团近似高斯分布的聚类，初始化的工作opencv好像已经帮住做了，可以忽略
input:  类别的个数，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testGMMImage -testImg=../data/39-1001150009181229150644600.png
```

#### testMeanShiftImage 聚类车道不合适
它的原理就是从一个点开始一直向密度较大的方向移动，每个点都作为初始点做一遍，重心重合度较大的重心归为一类，在计算密度的时候只考虑周围邻近[高斯加权]的像素，不是考虑所有的像素，每个聚类结束的条件是重心不移动了。
input:  计算密度的半径，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testMeanShiftImage -testImg=../data/39-1001150009181229150644600.png
```

#### testKMeansImage 聚类车道不合适
适合团状的数据，不适合车道
input:  类别，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testKMeansImage -testImg=../data/39-1001150009181229150644600.png
```

## TODO next
+ testDbScan中，读取测试数据时，强制将z坐标赋值为0,因为显示时只能二维显示，否则不好看出聚类结果
## Coding Reference
+ [dbScan](https://github.com/james-yoo/DBSCAN)
+ testDbScanFastImageDeSai可以不看，完全是为德赛准备的代码
