# 3MLWCDA


## 如何使用
### 配置
1. 拉取github的内容并安装3ML等相应环境:
`conda env create -f environment.yaml`

2. 将Env中的文件夹拷贝到对应环境的site-packages中

### 学习
3ML: https://threeml.readthedocs.io/ (重点看HAWC的插件)   https://github.com/threeML/threeML   https://github.com/threeML/hawc_hal

astromodels: https://astromodels.readthedocs.io/en/latest/

用 3MLnew_Crab.ipynb 或 3MLnew_Crab_jfit.ipynb 进行尝试

### 数据
 LHAASO合作组成员可以联系 caowy@mail.ustc.edu.cn

### 反馈
欢迎反映需求和bug, 以及通过github提交你的修改和新增内容, 我们共同开发.

## 更新
### V1.2更新 2024.1.16
* Myspeedup 创建
  * runllhskymap() 调用PBS加速获取llhskymap,大概6-10min一张6度roi天图,加速空间还很大,如合理分配节点以最大化利用多进程
* Mysigmap 更新
  * getllhskymap() 收集PBS结果并保存healpix和普通fits文件, 并画图
  * write_resmap() 更新 可设置自动AddUserInfo并提交残差天图的runllhskymap()
  * getsig1D()     单独的画一位显著性分布的函数
  * heal2fits()    Bug 修复
* Myfit 更新
  * get_sources()  无需依赖拟合result, 可以画初始化模型
  * Search()       更新 支持从catalog出发搜索新源, 支持KM2A
  * getcatModel()  更新 修复Bug
* Mymap 更新
  * maskroi()      Mask healpix 非roi区域

* Tools 更新
  * LHAASO trans
  * LHAASO llhskymap

### V1.1更新 2024.1.10
* Myfit 更新
  * jointfit() 联合拟合多个探测器
  * getcatModel() 获取本区域LHAASO catalog模型
* KM2A 最新天图及响应!

### V1.0更新 2024.1.1
* Mycatalog 更新
  * getViziercat()
  * GetAnyCat() transfer pands dataframe to what we need
  * drawcatsimple() Draw catalog and compare it with LHAASO
* Mycoord 更新常见坐标计算
  * change_coord() 更改healpix map坐标
* Myfit 更新
  * Search() 迭代搜源
  * set_diffusebkg() 自动计算当前区域弥散模版
* Mymap 更新
  * Draw_diffuse() 画出 diffuse 模版 contour 进行对比
  * hpDraw() 可指定方形区域skyrange参数, 如:画全天天图
  * maskdisk() maskdiskout() Mask 天图
  * maskroi() 生成带mask的ROI
* Mysigmap 更新
  * Draw_ellipse() 画指定坐标系椭圆
  * drawfits() 画fits文件或者透明叠加于传入fig
  * heal2fits() 截取healpix天图转fits格式
* Myspectrum 更新
  * Draw_spectrum_fromfile() 从生成的txt文件画能谱点


### V0.92更新
  横分布函数: Draw_lateral_distribution(map, ra, dec, num, width, ifdraw=False)

### V0.91更新
* 加入了2D 的halo template:Continuous_injection_diffusion2D(),以及对应的椭圆的Continuous_injection_diffusion_ellipse2D()
* 以及画catalog的方式
* 加了部分函数注释