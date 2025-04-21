"""
Created by lxy on 2024/9/29
"""
import sys
import time
import numpy as np
from numba import jit
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
from pathlib import Path
import csv
import logging
@jit(nopython=True)
def cal_rmse_mae(test_list,W,P,Qi,a,b):
    mae = 0
    rmse = 0
    for data in test_list:
        u = int(data[0])
        i = int(data[1])
        y = data[2]

        temp = np.dot(P[u, :], Qi[i, :])
        U = sigmoid(W * temp)

        y_pred = sigmoid(U + a[u] + b[i])

        mae += np.abs(y-y_pred)
        rmse += (y-y_pred) ** 2

    mae = mae / len(test_list)
    rmse = np.sqrt(rmse / len(test_list))

    return rmse,mae

@jit(nopython=True)
def sigmoid(a):
    return 1 / (1+np.exp(-a))

@jit(nopython=True)
def training(train_list,f,W,P,Qi,a,b,Va,Vb,Vw,Vp,Vq,momentum,lr,lamda):
    for data in train_list:
        u = int(data[0])
        i = int(data[1])
        y = data[2]

        temp = np.dot(P[u,:],Qi[i,:])

        U = sigmoid(W*temp)

        y_pred = sigmoid(U + a[u] + b[i])

        temp1 = (y_pred - y) * y_pred * (1 - y_pred)

        theta_ijk = (temp1 * U * (1 - U))
        Va[u] = momentum * Va[u] + lr * (temp1 + lamda * a[u])
        a[u] -= Va[u]
        Vb[i] = momentum * Vb[i] + lr * (temp1 + lamda * b[i])
        b[i] -= Vb[i]
        Vw = momentum * Vw + lr * (theta_ijk * temp + lamda * W)
        W -= Vw

        for r in range(f):
            Vp[u, r] = momentum*Vp[u,r] + lr*(theta_ijk*W*Qi[i,r]+lamda*P[u,r])
            P[u,r] -= Vp[u,r]

            Vq[i, r] = momentum*Vq[i,r] + lr*(theta_ijk*W*P[u,r]+lamda*Qi[i,r])
            Qi[i,r] -= Vq[i,r]

class NNLFA():
    def __init__(self,
                 train_list,
                 test_list,
                 f,
                 N,
                 I,
                 momentum,
                 lr,
                 lamda
                 ):
        self.train_list = train_list
        self.test_list = test_list
        self.f = f
        self.N = N
        self.I = I
        self.momentum = momentum
        self.lr = lr
        self.lamda = lamda
        self.convergence_round = 1000  # 收敛时的轮数
        self.flag_rmse = False  # true表示此时RMSE达到最小
        self.flag_mae = False
        self.min_rmse = 100  # 记录最小RMSE
        self.min_mae = 100
        self.min_rmse_round = 0  # 记录获取RMSE最小轮数
        self.min_mae_round = 0
        self.delay_count = 5 # 最小收敛轮数
        self.rand_min = -0.05
        self.rand_max = 0.05
        self.maximum_round_count = 1000

    def init_matrix(self):
        #LFT后的矩阵
        self.P = np.random.uniform(low=self.rand_min, high=self.rand_max, size=(self.N,self.f))
        self.Qi = np.random.uniform(low=self.rand_min, high=self.rand_max, size=(self.I, self.f))
        #权重矩阵
        self.W = np.random.randn()
        #偏置向量
        self.a = np.random.uniform(low=self.rand_min, high=self.rand_max, size=self.N)
        self.b = np.random.uniform(low=self.rand_min, high=self.rand_max, size=self.I)
        #速度矩阵
        self.Va = np.zeros(self.N)
        self.Vb = np.zeros(self.I)
        self.Vw = 0
        self.Vp = np.zeros(shape=(self.N,self.f))
        self.Vq = np.zeros(shape=(self.I,self.f))

    def train(self):

        train_start_time = time.time()  # 训练开始时间
        self.every_round_RMSE = np.full(self.maximum_round_count, float("inf"), dtype=np.float64)
        self.every_round_MAE = np.full(self.maximum_round_count, float("inf"), dtype=np.float64)

        records_list = []
        self.init_matrix()
        for i in range(self.maximum_round_count):
            inner_start_time = time.time()
            training(self.train_list, self.f, self.W, self.P, self.Qi, self.a,self.b, self.Va, self.Vb, self.Vw, self.Vp,self.Vq,self.momentum,self.lr,self.lamda)
            rmse, mae = cal_rmse_mae(self.test_list, self.W,self.P,self.Qi,self.a,self.b)
            self.every_round_RMSE[i] = rmse
            self.every_round_MAE[i] = mae
            records_list.append(np.array([rmse, mae]))
            logger.info(f"--------------------  Round {i}  --------------------")
            logger.info(f"当前RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
            if np.abs(self.every_round_RMSE[i] - self.min_rmse) >= 0.00001 and self.every_round_RMSE[i] < self.min_rmse:
                self.min_rmse = self.every_round_RMSE[i]
                self.min_mae = self.every_round_MAE[i]
                self.min_rmse_round = i
                self.min_mae_round = i
            elif i - self.min_rmse_round >= self.delay_count:
                break
            elif rmse > self.min_rmse:
                break
            self.convergence_round = i
            logger.info(f'当前最小的RMSE:{self.min_rmse:.4f} | 最小的MAE:{self.min_mae:.4f} | 轮数为:{self.min_rmse_round}')
            inner_end_time = time.time()
            logger.info(f'此轮的训练时间为:{(inner_end_time-inner_start_time):.2f}')
        train_end_time = time.time()
        time_sub = train_end_time - train_start_time
        print("Total time: ", time_sub)
        logger.info("--------------------  Training Completed  --------------------")
        logger.info(f'总训练时间为:{time_sub:.2f}')
        logger.info(f'最终收敛轮数为:{self.convergence_round}')
        logger.info(f"最小 RMSE: {self.min_rmse:.4f}  |  轮数: {self.min_rmse_round}")
        logger.info(f"最小 MAE: {self.min_mae:.4f}  |  轮数: {self.min_mae_round}")

        return self.min_rmse,self.min_mae,time_sub,self.convergence_round

if __name__ == '__main__':
    Model_name = 'NNLFA'
    # 获取当前脚本的绝对路径
    current_file_path = Path(__file__).resolve()
    # 找到项目的根目录，例如根目录是脚本的上两级目录
    BASE_DIR = current_file_path.parent.parent
    # 将项目根目录添加到系统路径中
    sys.path.append(str(BASE_DIR))

    # Define the list of datasets to process
    datasets = ['Flixster', 'EachMovie', 'big_anime', 'film_trust', 'ratings_disposition', 'Netflix']
    train_size = input("\nplease choose a train_size: 0.2, 0.4, 0.6, 0.8 :")
    train_size = float(train_size)
    DATASET = input("\nplease choose a dataset: ['Flixster,EachMovie,big_anime, film_trust, ratings_disposition,Netflix']: ")
    if DATASET not in datasets:
        warnings.warn(f"{DATASET} does not exist.")
    folder_path = f"{BASE_DIR}/data/{DATASET}/{Model_name}/model_log/log/"

    # 打开文件夹之前,确保文件夹存在,不存在的话创建文件夹
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # CSV文件的文件名
    csv_filename = f"{folder_path}_result.csv"
    # 如果文件不存在，则创建并写入表头
    if not Path(csv_filename).exists():
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DATASET", "gamma", "lr", "lambda", "rmse","mae","time","iteration"])

    para_config = pd.read_excel(f"{BASE_DIR}/para_config/{int(round(train_size*10))}/{DATASET}.xlsx",names=['gamma', 'lr', 'lambda'])
    dataset_config = pd.read_csv(f"{BASE_DIR}/data/{DATASET}/config.csv",header=0)

    # 创建日志文件路径
    log_filename = f"{folder_path}train_{int(round(train_size * 10))}_detail_logging.txt"
    # 创建日志记录器
    logger = logging.getLogger(__name__)  # 获取一个logger实例
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建一个文件处理器并设置编码为 utf-8
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 设置控制台处理器的级别

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)  # 应用格式到文件处理器
    console_handler.setFormatter(formatter)  # 应用格式到控制台处理器

    # 将处理器添加到logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    logger.info(f"Processing dataset: {DATASET}")
    logger.info(f"model_name: {Model_name}")

    # Load the dataset
    path = f"{BASE_DIR}/data/{DATASET}/{DATASET}_0_1.txt"
    data = pd.read_csv(path, names=['user_id', 'item_id', 'rating'], usecols=['user_id', 'item_id', 'rating'],header=0)

    # Calculate N (user count) and M (item count)
    num_user = int(dataset_config['num_user'][0])
    num_item = int(dataset_config['num_item'][0])

    # Split the data into train and test sets
    train_list, test_list = train_test_split(data, test_size=1- train_size, random_state=42)
    train_list = np.array(train_list)
    test_list = np.array(test_list)

    # para_setting
    gamma = float(para_config['gamma'])
    lr = float(para_config['lr'])
    lamda = float(para_config['lambda'])
    f = 20
    logger.info(f"train_ratio :{train_size}")
    logger.info(f"optimizer_params: gamma={gamma}, lr={lr}, lamda={lamda}")
    logger.info(f"dataset_detail: num_user={num_user}, num_item={num_item}")
    model = NNLFA(train_list, test_list,f,num_user,num_item,gamma,lr,lamda)
    rmse,mae,t,it = model.train()
    # Write results to CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([DATASET, gamma, lr, lamda, rmse,mae, t, it])
