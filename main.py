import numpy as np
import pandas as pd

# 读取训练数据集
training_raw_data = pd.read_csv("./data/training_set.csv", usecols=["AT", "EV", "AP", "RH", "PE"], encoding="utf-8").dropna().values

