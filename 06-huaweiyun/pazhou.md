```sh
sudo ssh -i /home/mw/project/pazhou_361 pazhou_361@172.18.3.12

# 数据地址
cd /public/pazhou/pazhou_data


```

## python环境

```sh
# conda create -n torch110_dtk2210 python==3.8

# 每次激活之前要先运行下述命令
module av  # 查看系统可用软件列表，执行命令。注意在超算登录节点运行

module purge # 加载前使用module purge清理环境
module list # 查看当前环境，运行 module purge 之后显示 No Modulefiles Currently Loaded.

# 加载需要的软件使用 module load 命令即可
module load anaconda3/5.2.0   #加载预置的anaconda3，以便正常使用conda命令
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/gcc-7.3.1
module load compiler/dtk/22.10  # dtk是dcu的驱动，可以理解为NVIDIA卡的cuda

source activate torch110_dtk2210

```

| ID    | liver-肝 | spleen-脾 | left kidney-左肾 | right kidney-右肾 |
| ----- | -------- | --------- | ---------------- | ----------------- |
| case1 | 1        | 0         | 0                | 0                 |
| case2 | 1        | 1         | 0                | 0                 |
| case3 | 0        | 0         | 1                | 0                 |
| case4 | 1        | 1         | 0                | 0                 |
| case5 | 1        | 1         | 0                | 0                 |
|       |          |           |                  |                   |



```python
import SimpleITK as sitk


path1 = './project/pazhou_data/train/data/case1.nii.gz'
NifitPath = path1
sitkImage = sitk.ReadImage(NifitPath)
# 查看图像的原点Origin，大小Size，间距Spacing和方向Direction
print("原点位置:{}".format(sitkImage.GetOrigin())) 
print("尺寸：{}".format(sitkImage.GetSize()))
print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()) )
print("图像方向:{}".format(sitkImage.GetDirection()))
# 查看图像相关的纬度信息
print("维度:{}".format(sitkImage.GetDimension()))
print("宽度:{}".format(sitkImage.GetWidth()))
print("高度:{}".format(sitkImage.GetHeight()))
print("深度(层数):{}".format(sitkImage.GetDepth()))
# 体素类型查询
print("数据类型:{}".format(sitkImage.GetPixelIDTypeAsString()))
```

### SimpleITK 与 Numpy 之间的转换

```python
# 注意SimpleITK图像转换前是【x,y,z】，转换后是【z,y,x】，两者刚好是反过来
print("创建一个3D SimpleITK图像")
sitkImage = sitk.Image([200,100,50], sitk.sitkUInt16)

#3D SimpleITK 转 Numpy
print("3D SimpleITK 转 Numpy")
npImage = sitk.GetArrayFromImage(sitkImage)
print("打印转换前sitkImage的Size(x,y,z):{}".format(sitkImage.GetSize()))
print("打印转换后npImage的Size(z,y,x):{}".format(npImage.shape))
print()
print("Numpy 转 SimpleITK")
sitkImage = sitk.GetImageFromArray(npImage)
print("打印转换前npImage的Size(z,y,x):{}".format(npImage.shape))
print("打印转换后sitkImage的Size(x,y,z):{}".format(sitkImage.GetSize()))

print('='*30+'我是分割线'+'='*30)

print("创建一个 RGB SimpleITK图像")
sitkImage= sitk.Image((512,512), sitk.sitkVectorUInt8, 3)

#RGB SimpleITK 转 Numpy
print("RGB SimpleITK 转 Numpy【注意size的变化】")
npImage = sitk.GetArrayFromImage(sitkImage)
print("打印转换前sitkImage的Size(x,y,z):{}".format(sitkImage.GetSize()))
print("打印转换后npImage的Size(z,y,x):{}".format(npImage.shape))
print()
print("Numpy 转 SimpleITK")
sitkImage = sitk.GetImageFromArray(npImage)
print("打印转换前npImage的Size(z,y,x):{}".format(npImage.shape))
print("打印转换后sitkImage的Size(x,y,z):{}".format(sitkImage.GetSize()))
```

### 可视化

```python
"""
有些医学图像用matplotlib 显示的时候会出现上下翻转的情况，但是用ITK——SNAP打开显示却是正常。
可能是matplotlib的显示原点不同的关系。遇到这个中期，就通过强制翻转一下。
"""
#读取MR图像
path1 = './project/pazhou_data/train/data/case1.nii.gz'
NifitmPath = path1

sitkImage = sitk.ReadImage(NifitmPath)

z = int(sitkImage.GetSize()[2]/2)
slice = sitk.GetArrayFromImage(sitkImage)[z,:,:]
plt.figure(figsize=(5,5))
plt.imshow(slice, 'gray')
plt.show()

slice = np.flipud(slice)
plt.figure(figsize=(5,5))
plt.imshow(slice, 'gray')
plt.show()
```

### 重采样-统一Spacing

```python
def resampleSpacing(sitkImage, newspace=(1,1,1)):
    """
    统一Spacing
    由于x轴和y轴的Spcing由小变大，x轴和y轴的size就变小。
    z轴的Spacing由大变小，z轴的size就变大
    """
    euler3d = sitk.Euler3DTransform()
    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #新的X轴的Size = 旧X轴的Size *（原X轴的Spacing / 新设定的Spacing）
    new_size = (int(xsize*xspacing/newspace[0]),int(ysize*yspacing/newspace[1]),int(zsize*zspacing/newspace[2]))
    #如果是对标签进行重采样，模式使用最近邻插值，避免增加不必要的像素值
    sitkImage = sitk.Resample(sitkImage,new_size,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage


#读取nifit原数据 ，size为：(880, 880, 12)
path1 = './project/pazhou_data/train/data/case1.nii.gz'
NifitmPath = path1
sitkImage = sitk.ReadImage(NifitmPath)
print("重采样前的信息") 
print("尺寸：{}".format(sitkImage.GetSize()))
print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()) )

print('='*30+'我是分割线'+'='*30)

newResample = resampleSpacing(sitkImage, newspace=[1,1,1])
print("重采样后的信息")
print("尺寸：{}".format(newResample.GetSize()))
print("体素大小(x,y,z):{}".format(newResample.GetSpacing()))
```

### 重采样-统一Size

```python
def resampleSize(sitkImage, depth):
    """
    统一Size
    X轴和Y轴的Size和Spacing没有变化，
    Z轴的Size和Spacing有变化
    """
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing/(depth/float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    #根据新的spacing 计算新的size
    newsize = (xsize,ysize,int(zsize*zspacing/new_spacing_z))
    newspace = (xspacing, yspacing, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
    return sitkImage


DEPTH = 16  # 需要重采样Size的层数

# 读取nifit原数据 ，size为：(880, 880, 12)
path1 = './project/pazhou_data/train/data/case1.nii.gz'
NifitmPath = path1
sitkImage = sitk.ReadImage(NifitmPath)
print("重采样前的信息") 
print("尺寸：{}".format(sitkImage.GetSize()))
print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()) )

print('='*30+'我是分割线'+'='*30)


newsitkImage = resampleSize(sitkImage, depth=DEPTH)
print("重采样后的信息")
print("尺寸：{}".format(newsitkImage.GetSize()))
print("体素大小(x,y,z):{}".format(newsitkImage.GetSpacing()) )
```

### 把预测的Mask转换成原数据一致信息的NiFit文件

```python
def print_info(sitkImage):
    print("原点位置:{}".format(sitkImage.GetOrigin())) 
    print("尺寸：{}".format(sitkImage.GetSize()))
    print("体素大小(x,y,z):{}".format(sitkImage.GetSpacing()) )
    print("图像方向:{}".format(sitkImage.GetDirection()))
    print("维度:{}".format(sitkImage.GetDimension()))
    print("宽度:{}".format(sitkImage.GetWidth()))
    print("高度:{}".format(sitkImage.GetHeight()))
    print("深度(层数):{}".format(sitkImage.GetDepth()))
    print("数据类型:{}".format(sitkImage.GetPixelIDTypeAsString()))
    print()

import numpy as np

#读取nifit原数据 ，size为：(880, 880, 12)
path1 = './project/pazhou_data/train/data/case1.nii.gz'
NifitmPath = path1
sitkImage = sitk.ReadImage(NifitmPath)

#创建numpy的数据 ，假设这个是模型预测的文件，shape为：(880, 880, 12)
shape = sitkImage.GetSize()
npImage = sitk.GetArrayFromImage(sitkImage)
shape = npImage.shape
npMask = np.zeros(shape, np.float32)

#现在把预测numpy数据转换成和原nifit数据一致的信息，数据类型是sitkUInt8
# 先转换成sitkImage
sitkMask = sitk.GetImageFromArray(npMask)
# 转换数据类型
sitkMask = sitk.Cast(sitkMask, sitk.sitkUInt8)


# 方式一
# 手动设置
sitkMask.SetSpacing(sitkImage.GetSpacing()) #设置spacing
sitkMask.SetOrigin(sitkImage.GetOrigin()) #设置 origin
sitkMask.SetDirection(sitkImage.GetDirection())  #设置方向
#打印两者的信息，除了数据类型，其他都一致
print_info(sitkMask)
print_info(sitkImage)
print('='*30+'我是分割线'+'='*30)

#方式二
#调用CopyInformation即可
sitkMask.CopyInformation(sitkImage)
#打印两者的信息，除了数据类型，其他都一致
print_info(sitkMask)
print_info(sitkImage)

#最后把预测的mask文件保存成NiFit数据即可
sitk.WriteImage(sitkMask,'predictMask.nii.gz')
```

