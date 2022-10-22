# 工作内容记录

| Time |           key-point           |                            Detail                            | Progress |
| :--: | :---------------------------: | :----------------------------------------------------------: | :------: |
| 1003 | 构建backbone、neck、yoloxhead | 修正完成backbone、neck、yoloxhead的model_size输出，并构建完成建立模型的过程，yoloxhead中包括了yolox loss计算，backbone包括darknet、cspdarknet，head主要包括了两种解耦方式的检测头 |    ✅     |
| 1004 |            完成exp            |           构建每个训练任务的模板，包括数据，模型等           |    ✅     |
| 1005 | 修正fpn为输入输出的通道数相同 |  均使用相同的model_size参数同一构建backbone、neck、head模块  |    ✅     |
| 1007 |           Trainer类           |             Trainer实现每个定义好的exp的训练过程             |    ✅     |
| 1014 |              DDP              |                       实现单机多卡训练                       |    ✅     |
| 1015 |   测试基于DIOR&nano版本代码   | 1007下午被挤掉 1008重新开始 任务内容记录在train_log.txt，目前已完成train和val |    ✅     |
| 1020 |     1、shell运行 2、TOOD      |                1、可以直接使用命令行运行代码                 |          |

# 2计划module

### Backbone

| Progress |   model    |             model_size              | Detail | Time |
| :------: | :--------: | :---------------------------------: | :----: | :--: |
|    ✅     |  darknet   |               21、53                |        | 1003 |
|    ✅     | cspdarknet |            n、s、m、l、x            |        | 1003 |
|    ✅     |   resnet   | res18、res34、res50、res101、res152 |        | 1010 |
|          |    vgg     |                                     |        |      |
|          |  connext   |                                     |        |      |
|          |    vit     |                                     |        |      |
|          |   repVGG   |                                     |        |      |



### Neck

本项目中实现的neck中的涉及到的每个layer的block数目都默认与backbone中第层中的block数目相同

| Progress | model | Detail | Time |
| :------: | :---: | :----: | :--: |
|    ✅     |  FPN  |        | 1003 |
|    ✅     |  PAN  |        | 1003 |
|          | CABM  |        |      |

### Head

| Progress |   model   |               Detail                | Time |
| :------: | :-------: | :---------------------------------: | :--: |
|    ✅     | Decoupled | 在YOLOX中的+修改的Feature Align方式 | 1003 |

## Label Assign

| Progress | model  |   Detail    | Time |
| :------: | :----: | :---------: | :--: |
|    ✅     | simOTA | YOLOX中使用 | 1007 |
|          |  ATSS  |             |      |
|          |  TOOD  |             |      |

## Demo

| Progress | 任务项 |   Detail   | Time |
| :------: | :----: | :--------: | :--: |
|          |  CAM   | 模型可视化 |      |

# 常用远程工具

查看端口是否被占用使用 `netstat` 命令 

visdom远程使用指南

```shell
# 1、ssh重定向到本地
ssh -L 18097:127.0.0.1:8097 username@ip
# 2、在服务器上使用指定端口正常启动visdom：	
python -m visdom.server -p 使用端口号
# 3、本地浏览器输入输入地址。
http://localhost:18097
```

[tensorboard远程](https://blog.csdn.net/weixin_35653315/article/details/71327740)

```shell
# 1、ssh重定向到本地
ssh -L 16006:127.0.0.1:6006 username@remote_server_ip

# 2、在服务器上使用6006端口正常启动tensorboard：	
tensorboard --logdir=xxx --port=6006

# 3、在本地浏览器中输入地址：
127.0.0.1:16006
```

断开ssh连接

```shell
pkill -kill -t pts/x
```



# 报错

1、AttributeError:module ‘distutils’ has no attribute 'version’

```shell
pip install --upgrade setuptools==56.1.0
```



2、[shell 找不到包](https://blog.csdn.net/pengchengliu/article/details/117752340?utm_term=linux%E8%BF%90%E8%A1%8Cpython%E4%BB%A3%E7%A0%81%E6%89%BE%E4%B8%8D%E5%88%B0%E5%8C%85&utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~sobaiduweb~default-2-117752340&spm=3001.4430)，在Terminal执行以下命令

```shell
export PYTHONPATH=/path/to/project
```

