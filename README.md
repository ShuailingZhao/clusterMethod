## 聚类方法dbscan

#### testDbScan 聚类团装点
input:  最小密度，半径，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testDbScan -testData=../data/benchmark_hepta.dat
```

#### testDbScanImage 聚类车道
input:  最小密度，半径，测试数据
output: 聚类结果
```
cd build
rm -rf *
cmake .. && make -j4
./testDbScanImage -testImg=../data/39-1001150009181229150644600.png
```

## TODO next
+ testDbScan中，读取测试数据时，强制将z坐标赋值为0,因为显示时只能二维显示，否则不好看出聚类结果
## Coding Reference
+ [dbScan](https://github.com/james-yoo/DBSCAN)
