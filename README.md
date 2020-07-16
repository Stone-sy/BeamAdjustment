BeamAdjustment
=================
2020年数字摄影测量实习--光束法平差

****
## 目录
* 使用方法
    * [直接运行exe文件](#直接运行exe文件)
    * [运行源代码](#运行源代码)
* 光束法平差内容
* 文件结构

使用方法
------

#### 直接运行exe文件 
    点击BeamAdjustment.exe，稍微等1分钟左右，弹框会显示程序运行进度。程序结果输出到Result文件夹中。
#### 运行源代码 
1. 安装环境依赖
   ```shell
   pip install -r requirements.txt
   ```

2. 运行代码
    ```shell
    python BeamAdjustment.py
    ```

光束法平差内容
------
#### 后方交会
    已知像点坐标和控制点坐标，利用像点坐标纠正、后方交会迭代，获得相片的外方位元素。
#### 前方交会
    已知相片的外方位元素和像点坐标，通过前方交会，获得加密点坐标。
#### 光束法平差
    已知相片的外方位元素、像点坐标、加密点坐标，以此为初值进行光束法平差的迭代。之所以进行后方交会和前方交会，就是为了获得更好的初值，不然迭代不收敛。光束法平差的结果是更加准确的相片的外方位元素和加密点坐标。
#### 精度评定
    将光束法平差的结果、理论精度、实际精度输出到Result文件夹中对应csv里。
#### 核线
    在相片002.jpg和相片004.jpg上绘制通过控制点4507点的同名核线。

文件结构
------
#### BeamAdjustment
> Data ：存放已知的像点坐标、控制点坐标和相片</br>
> Result ：存放光束法平差输出结果、精度评定和核线绘制结果</br>
> BeamAdjustment.py ：源代码</br>
> BeamAdjustment.exe ：可执行程序</br>
> requirements.txt ：所需python包及版本</br>
> Readme.md ：使用须知</br>
