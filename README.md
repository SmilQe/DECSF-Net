# DECSF-Net
### DECSF‑Net: a multi‑variable prediction method for pond aquaculture water quality based on cross‑source feedback fusion
This is the official PyTorch implementation of our paper: "[**DECSF‑Net: a multi‑variable prediction method for pond aquaculture water quality based on cross‑source feedback fusion
**](https://link.springer.com/article/10.1007/s10499-025-02005-9)". If you find our work useful, we'd appreciate you citing our paper as follows:

```
@article{song2025decsf,
  title={DECSF-Net: a multi-variable prediction method for pond aquaculture water quality based on cross-source feedback fusion},
  author={Song, Liqiao and Song, Yizhong and Tian, Yunchen and Quan, Jianing},
  journal={Aquaculture International},
  volume={33},
  number={4},
  pages={1--25},
  year={2025},
  publisher={Springer}
}
```

## Introduction
Traditional pond aquaculture water quality prediction models have limitations when processing cross-source heterogeneous data, particularly due to their failure to fully account for the impact of future meteorological data on water quality changes. Meteorological factors like temperature, air pressure, and rainfall typically cause significant lag effects on water quality changes. Existing models often rely solely on historical water quality data for predictions, overlooking the influence of meteorological factors. This paper proposes an enhanced deep learning model, the dual encoder cross-source feedback network (DECSF-Net), incorporating a modified dual encoder structure to encode water quality and meteorological data separately. This design accurately captures the complex impact of future meteorological data on water quality time series. The cross-source feedback fusion (CSFF) module enhances mutual attention between water quality and meteorological data through a bidirectional feedback mechanism, improving the model's ability to jointly represent cross-source data. Experimental results demonstrate that DECSF-Net outperforms existing mainstream methods in predicting water quality for the next 8 hours, with a mean squared error (MSE) of 0.0959, mean absolute error (MAE) of 0.2037, root mean squared error (RMSE) of 0.3084, and mean absolute percentage error (MAPE) of 1.5159, showcasing its superior prediction accuracy. This model effectively addresses the water quality prediction challenges in complex ecological environments. The paper shows that integrating future meteorological data into water quality prediction methods significantly improves accuracy, offering substantial practical value.

## Usage
### Dependencies
We recommend using [**anaconda**](https://www.anaconda.com/) for managing dependency and environments.

模型名称: DECSF-Net, 代码中使用名为 iMMTST, iMMTST 等同于 DECSF-Net

水质数据集放在了 ./dataset/fill.csv

水质模型的训练脚本放在了 ./train.sh, 在终端执行 bash ./train.sh 开始训练模型

result.txt 存放 模型在测试集上的指标

log_imm 存放 模型训练日志

models 存放模型代码

checkpoints 存放模型权重

tf_logs 是 tensorboard 训练记录

model weights：[**Google**]([https://www.anaconda.com/https://drive.google.com/file/d/1VhjCpHJ6Z2YrLazz4DWETq7AMnPmkjPj/view?usp=drive_link])
