import numpy as np
from scipy.special import erfinv
from Environment.EnvUtils import *
from scipy.optimize import linear_sum_assignment
import pandas as pd


# Collaborative Perception Application  协同感知业务  缩写 CPA
# Collaborative Control Application   协同控制业务  简称：CTA

class Users(object):
    def __init__(self):
        self.dist = None
        self.pathloss = None
        self.channel_gain = None


class CPAs(Users):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.rate_all = 0.0  # 50ms
        self.rate_slot = 0.0
        self.rate = None
        self.idx_Vehicles = None  # 发送感知数据的车
        self.avg_rate = 0.0
        self.pow = 0
        self.SINR = 0
        self.if_P_NOMA = []  # 叠加中的非正交多址技术  if_P_NOMA=1 为PD-NOMA  if_S_NOMA=1为SD-NOMA   都为0 且 partner不为NONE    则为   punching技术
        self.if_S_NOMA = []
        self.SC_idx = []  # subchannel
        self.partner = []


class CTAs(Users):
    def __init__(self):
        super().__init__()
        self.id = None
        self.pow = None
        self.rate = None
        self.SINR = None
        self.idx_Vehicles = None  # 发送和接收感知数据的车  例如[0，1]
        self.if_P_NOMA = 0
        self.if_S_NOMA = 0
        self.SC_idx = None
        self.partner = None


# 协同控制车辆的协作方式
class CTAs_all(Users):
    def __init__(self):
        super().__init__()
        self.dist = None
        self.TX_idx = None
        self.RX_idx = None


class Slidingwindow(object):
    def __init__(self, length, his=3):
        self.start_time: float = 0.0
        self.num_CTA = deque(maxlen=length * 350)
        self.CPA_delay_dist = deque(maxlen=1)  # 协同感知的时延，用于计算QOS
        self.CTA_pls_all = deque(maxlen=length * 350)
        self.SINR_CTA = deque(maxlen=10000)
        self.SINR_CPA = deque(maxlen=6)
        self.complexity = 0

        self.avg_gain_CP = []
        self.avg_gain_CC = []
        self.avg_gain_CP_CC = []
        self.avg_gain_CC_CP = []
        self.CPA_pls = 0
        self.CPA_Synchronicity = 0


class Vehicles(object):
    def __init__(self):
        self.position = None
        self.speed = None
        self.id = None


class NS_Env:
    def __init__(self, args: Dict[str, Any]) -> None:

        self.complexity_max = 3
        self.max = 0.0
        self.CPA_rate = []
        self.qos_CP = 0
        self.qos_CC = 0
        self.qos_complexity = 0
        self.complexity: float = 0.0
        self.CTA_pls = 0
        # 信道状态，目前只添加了平均pathloss
        self.min_gain_CC: float = 0.0
        self.min_gain_CP: float = 0.0
        self.max_gain_CC: float = 0.0
        self.max_gain_CP: float = 0.0
        self.avg_gain_CC: float = 0.0
        self.avg_gain_CP: float = 0.0
        #
        # 信道条件
        self.CPA_CTA_pathloss = []  # 感知对控制车辆的pathloss
        self.CPA_CTA_CSI = []
        self.CTA_CPA_pathloss = []  # 控制对感知车辆的pathloss
        self.CTA_CPA_CSI = []
        self.CTA_pathloss = []
        self.CTA_CSI = []
        #
        #
        self.low_policy = args['low_policy']
        #
        self.length = args['windows_length']
        self.windows = None
        self.CPA = None
        self.CTA = None
        self.reward: float = 0.0

        self.sys_SC_bandwidth = 1.44 * 10 ** 6  # 子载波带宽
        self.tx_bandwidth = 2 * 10 ** 7
        self.noise_density_dB = -150  # 噪声
        self.Nr = 1  # 接收天线
        self.Nt = 1
        self.num_of_CPA = 5  # 这里生成的应该是CPA的用户
        self.num_of_CTA = None

        self.max_slot = args["max_steps"]
        self.size_of_CTA = 180  # CTA 的数据包大小 byte
        self.size_of_CPA = 80000  # 200ms 0.2M
        self.au: float = 0.0  # CTA的激活概率

        self.num_of_SCs = 10  # 子载波个数
        self.SCs_of_CPA_pro = 3
        self.SCs_share = 6
        self.SCs_of_CTA = 1
        self.SCs_of_CPA = self.SCs_of_CPA_pro + self.SCs_share

        self.Rician_fading_factor = 9  # 瑞丽衰落
        self.sinr_th = 10 ** 2.15  # CTA最小SINR
        self.SINR_max_CPA = 10 ** 3  # 为30db
        self.SINR_min_CPA = 10 ** 0.6

        self.slot_time: float = 10 ** -3  # 1ms
        self.M = 7  # 一个时隙处的微时隙个数
        self.mini_time = self.slot_time / self.M
        self.rate_max = self.sys_SC_bandwidth * np.log2(
            1 + self.SINR_max_CPA) * self.mini_time

        self.error_rate: float = 0.0001  # CTA丢包率
        self.Rb_max = 200  # 23dbm
        self.noise_power = self.sys_SC_bandwidth * 10 ** (self.noise_density_dB / 10)  # mw
        self.Scs_UE = [100 for _ in range(self.num_of_SCs)]  # 协同感知的分配情况，用于objective function计算和用户匹配
        self.CP_cav = None  # 主车辆
        self.CC_cav = None  # 辅助车辆
        self.CP_num = 5  # 正在进行感知的车辆数量
        self.CC_num = 12
        self.bs_position: list[float] = [300, 70, 25]
        self.CTA_all = [CTAs_all() for i in range(self.CC_num - 2)]  # 修改
        self.V2V_shadow_db = 3  # 标准差
        self.avg_rate = [0 for _ in range(self.num_of_CPA)]
        self.max_rate = [0 for _ in range(self.num_of_CPA)]

        self.act_CC_num = [0 for _ in range(self.CC_num - 2)]  # 一个slot中每对协同控制的车辆的激活数量
        self.pls_CC_num = [0 for _ in range(self.CC_num - 2)]  # 一个slot中每对协同控制的车辆的丢包数

        self.hist_CP_g = np.zeros((self.num_of_CPA, self.num_of_SCs), dtype=complex)
        self.hist_CC_g = np.zeros((self.CC_num - 2, self.num_of_SCs), dtype=complex)
        self.hist_CP_CC_g = np.zeros((self.CP_num, self.CC_num, self.num_of_SCs), dtype=complex)
        self.hist_CC_CP_g = np.zeros((self.CC_num, self.num_of_SCs), dtype=complex)

        # 分别为打孔 PD-NOMA SD-NOMA的网络复杂度系数
        self.alpha0: float = 0.01
        self.alpha1: float = 1.0
        # reward
        self.w1 = 1 / 5
        self.w2 = 60
        self.w3 = 0.3
        self.w4 = 50
        self.w5 = 0.75
        self.done = 0
        # ------------------------ option PART
        self.k1: float = 0.0
        self.k2: float = 0.0
        self.k3: float = 0.0
        self.k4: float = 0.0
        self.beta = 0.05

        self.CC_transmission_rate_CP: float = 10
        self.CC_transmission_rate_CC: float = 0.1
        self.CP_packet_loss_CP: float = -1.0
        self.CP_packet_loss_CC: float = -5.0
        self.avg_CP = 0.3
        self.avg_CC = 0.001
        self.complexity_CP = -1
        self.complexity_CC = -10


        self.k1 = self.CC_transmission_rate_CC
        self.k2 = self.CP_packet_loss_CC
        self.k3 = self.avg_CC
        self.k4 = self.complexity_CC
        self.beta = 0.05
        self.option = 0


        all_seed()

    def get_vehicle_position(self):
        df = pd.read_csv('./Environment/vehicles.csv')
        df = df[df['time'] == self.windows.start_time / 1000]

        CP_cav = [Vehicles() for _ in range(self.CP_num - 1)]
        CC_cav = [Vehicles() for _ in range(self.CC_num)]
        #
        e_idx = 0
        a_idx = 0
        for _, row in df.iterrows():
            vehicle_type = row['type']
            if vehicle_type == "E_CAV" and e_idx < self.CP_num - 1:
                CP_cav[e_idx].position = [row['x'], row['y'], 1.5]
                CP_cav[e_idx].speed = row['speed']
                CP_cav[e_idx].speed = row['id']
                e_idx += 1
            elif vehicle_type == "A_CAV" and a_idx < self.CC_num:
                CC_cav[a_idx].position = [row['x'], row['y'], 1.5]
                CC_cav[a_idx].speed = row['speed']
                CC_cav[a_idx].id = row['id']
                a_idx += 1
        return CP_cav, CC_cav

    def calculate_positon(self, p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return math.hypot(dx, dy, dz)

    # 小尺度衰落
    def user_CSI(self):
        x = np.sqrt(self.Rician_fading_factor / (self.Rician_fading_factor + 1))  # 小尺度衰落因子
        y = np.sqrt(1 / (self.Rician_fading_factor + 1))

        #
        for i in range(self.num_of_CPA):
            self.CPA[i].rate_slot = 0
            channel_gain = np.zeros(self.num_of_SCs, dtype=complex)  # 不再用 (Nt, num_of_SCs)
            # === 向量化生成 num_of_SCs 个子载波的信道增益 ===
            real_randn = np.random.randn(self.num_of_SCs)
            imag_randn = np.random.randn(self.num_of_SCs)
            fading = (x + y * np.sqrt(0.5) * (real_randn + 1j * imag_randn)
                      )
            h = np.sqrt(self.CPA[i].pathloss) * fading
            gain = np.abs(h) ** 2
            self.windows.avg_gain_CP[i] = gain.real
            self.CPA[i].channel_gain = gain.real

        # 控制的信道增益
        real_randn = np.random.randn(self.CC_num - 2, self.num_of_SCs)
        imag_randn = np.random.randn(self.CC_num - 2, self.num_of_SCs)
        fading = (
                x + y * np.sqrt(0.5) * (real_randn + 1j * imag_randn)
        )
        pathloss_array = np.sqrt(np.array(self.CTA_pathloss)).reshape(-1, 1)
        h = fading * pathloss_array  # shape: (num_CTA, num_SC)
        gain = np.abs(h) ** 2
        self.CTA_CSI = gain.real

        # 控制对感知的干扰
        real_randn = np.random.randn(self.CC_num, self.num_of_SCs)
        imag_randn = np.random.randn(self.CC_num, self.num_of_SCs)

        fading = (
                x + y * np.sqrt(0.5) * (real_randn + 1j * imag_randn)
        )
        pathloss_array = np.sqrt(np.array(self.CTA_CPA_pathloss)).reshape(-1, 1)
        h = pathloss_array * fading
        gain = np.abs(h) ** 2
        self.CTA_CPA_CSI = gain.real
        #

        # 感知对控制的干扰
        CP_num = self.CP_num
        CC_num = self.CC_num
        num_SC = self.num_of_SCs
        pathloss_matrix = np.sqrt(np.array(self.CPA_CTA_pathloss))  # shape: (CP_num, CC_num)
        real_randn = np.random.randn(CP_num, CC_num, num_SC)
        imag_randn = np.random.randn(CP_num, CC_num, num_SC)
        fading = (
                x + y * np.sqrt(0.5) * (real_randn + 1j * imag_randn)
        )
        h = fading * pathloss_matrix[:, :, None]  # shape: (CP_num, CC_num, num_SC)
        gain = np.abs(h) ** 2
        self.CPA_CTA_CSI = gain.real

    def get_user_pathloss(self):
        # 计算感知用户的pathloss
        self.CPA = [CPAs() for _ in range(self.num_of_CPA)]  # 给定CPA速率要求
        for i in range(self.num_of_CPA):
            if i < self.num_of_CPA - 1:
                dist = self.calculate_positon(self.CP_cav[i].position, self.bs_position)
            else:
                dist = self.calculate_positon(self.CC_cav[self.CC_num - 1].position, self.bs_position)  # 头车
            self.CPA[i].dist = dist
            # UMA:pathloss = 28.0 + 22 * np.log10(dist) + 20 * np.log10(5.9) + np.random.normal(0, 4)
            # RMA:pathloss = 20 * np.log10(40*np.pi*dist*5.9/3)+7.61 *np.log10(dist)- 11.1 + 0.002*np.log10(25)*dist
            pathloss = 20 * np.log10(247.13 * dist) + 7.61 * np.log10(dist) - 11.1 + 0.0028 * dist + np.random.normal(0,
                                                                                                                      4)
            self.CPA[i].pathloss = 10 ** (-pathloss / 10)
            self.CPA[i].idx_Vehicles = i

        # 协同控制车辆之间的pathloss
        self.CTA_pathloss = []
        for i in range(self.CC_num - 2):
            dist = self.CTA_all[i].dist
            pathloss = 32.4 + 20 * math.log10(dist) + 20 * math.log10(5.9)  # fc
            loss = pathloss + np.random.normal(0, self.V2V_shadow_db)
            P_los = min(1, 1.05 * np.exp(-0.0114 * dist))
            if np.random.rand() < P_los:
                additional_loss = 0
            else:
                additional_loss = 5 + max(0, 15 + np.log10(dist) - 41)
            loss += additional_loss
            channel_gain = 10 ** (-loss / 10)
            self.CTA_pathloss.append(channel_gain)

        self.CPA_CTA_pathloss = []
        # 计算感知对控制的干扰pathloss
        for i in range(self.CP_num):
            a = []
            for j in range(self.CC_num):
                if i < self.CP_num - 1:
                    dist = self.calculate_positon(self.CP_cav[i].position, self.CC_cav[j].position)
                else:
                    dist = self.calculate_positon(self.CC_cav[self.CC_num - 1].position,
                                                  self.CC_cav[j].position)  # 头车
                if i == self.CP_num - 1 and j == self.CC_num - 1:
                    a.append(0)
                    continue
                pathloss = 32.4 + 20 * math.log10(dist) + 20 * math.log10(5.9)  # fc
                loss = pathloss + np.random.normal(0, self.V2V_shadow_db)
                P_los = min(1, 1.05 * np.exp(-0.0114 * dist))

                if np.random.rand() < P_los:
                    additional_loss = 0
                else:
                    additional_loss = 5 + max(0, 15 + np.log10(dist) - 41)
                loss += additional_loss
                path_loss = 10 ** (-loss / 10)
                a.append(path_loss)
            self.CPA_CTA_pathloss.append(a)

        self.CTA_CPA_pathloss = []
        # 计算控制对感知的干扰pathloss
        for i in range(self.CC_num):
            dist = self.calculate_positon(self.CC_cav[i].position, self.bs_position)
            pathloss = 20 * np.log10(247.13 * dist) + 7.61 * np.log10(dist) - 11.1 + 0.0028 * dist + np.random.normal(0,
                                                                                                                      4)  # 6db
            pathloss = 10 ** (-pathloss / 10)
            self.CTA_CPA_pathloss.append(pathloss)

    def get_CTA_all(self):
        idx_vehicles = 0
        for i in range(self.CC_num // 2):
            self.CTA_all[idx_vehicles].dist = self.calculate_positon(
                self.CC_cav[self.CC_num - 1 - 2 * i].position, self.CC_cav[self.CC_num - 2 - 2 * i].position)
            self.CTA_all[idx_vehicles].TX_idx = self.CC_num - 1 - 2 * i
            self.CTA_all[idx_vehicles].RX_idx = self.CC_num - 2 - 2 * i
            idx_vehicles += 1

        self.CTA_all[idx_vehicles].dist = self.calculate_positon(
            self.CC_cav[0].position, self.CC_cav[4].position)
        self.CTA_all[idx_vehicles].TX_idx = 0
        self.CTA_all[idx_vehicles].RX_idx = 4
        idx_vehicles += 1

        self.CTA_all[idx_vehicles].dist = self.calculate_positon(
            self.CC_cav[0].position, self.CC_cav[5].position)
        self.CTA_all[idx_vehicles].TX_idx = 0
        self.CTA_all[idx_vehicles].RX_idx = 5
        idx_vehicles += 1

        self.CTA_all[idx_vehicles].dist = self.calculate_positon(
            self.CC_cav[10].position, self.CC_cav[2].position)
        self.CTA_all[idx_vehicles].TX_idx = 10
        self.CTA_all[idx_vehicles].RX_idx = 2
        idx_vehicles += 1

        self.CTA_all[idx_vehicles].dist = self.calculate_positon(
            self.CC_cav[10].position, self.CC_cav[3].position)
        self.CTA_all[idx_vehicles].TX_idx = 10
        self.CTA_all[idx_vehicles].RX_idx = 3

    def CTA_traffic_Pattern(self, step):
        t = step * 0.05  # 第几个long time
        T = 6
        A = 0.11  # 振幅
        B = 0.05  # 起始值（偏置）
        P = A * np.sin(np.pi * t / T) + B
        self.au = np.clip(P, 0.05, 0.3)

        self.au_all += self.au

    def CTA_traffic(self):
        self.num_of_CTA = 0
        self.CTA = []
        id = 0
        for i in range(self.CC_num - 2):
            if np.random.rand() < self.au:
                self.CTA.append(CTAs())
                self.num_of_CTA += 1
                self.CTA[id].channel_gain = self.CTA_CSI[i]
                self.CTA[id].idx_Vehicles = [self.CTA_all[i].TX_idx, self.CTA_all[i].RX_idx]
                self.CTA[id].id = i
                id += 1
            else:
                continue
        self.windows.num_CTA.append(self.num_of_CTA)

    # def get_avg_gain(self):
    #
    #     for i in range(self.CP_num):
    #         self.windows.avg_gain_CP.append(np.average(self.CPA[i].channel_gain) / np.double(1e-9) - 1)
    #     # self.avg_gain_CP = np.average(all)
    #     # self.min_gain_CP = np.min(all)
    #     # self.max_gain_CP = np.max(all)
    #
    #     for i in range(self.CC_num - 2):
    #         self.windows.avg_gain_CC.append(np.average(np.average(self.CTA_CSI[i]) / np.double(1e-8) - 1))
    #
    #     # self.avg_gain_CC = np.average(all)
    #     # self.min_gain_CC = np.min(all)
    #     # self.max_gain_CC = np.max(all)
    #
    #     for i in range(self.CC_num):
    #         self.windows.avg_gain_CC_CP.append(np.average(np.average(self.CTA_CPA_CSI) / np.double(1e-9) - 1))
    #     #
    #     # 感知对控制的干扰
    #     for i in range(self.CP_num):
    #         for j in range(self.CC_num):
    #             self.windows.avg_gain_CP_CC.append(np.average(self.CTA_CPA_CSI[i][j]))
    def get_avg_pathloss(self):
        all = []
        for i in range(self.CP_num):
            all.append(self.CPA[i].pathloss)
        self.avg_pathloss_CP = all
        all = []
        for i in range(self.CC_num - 2):
            all.append(self.CTA_pathloss[i])
        self.avg_pathloss_CC = all

    def schedule_CPA(self, max_SCs_per_user=None):
        """
        PF调度算法：每个用户可以不分配，SC只能分配给一个用户
        :param max_SCs_per_user: 每个用户最多可占用SC数（可选，None表示不限）
        """
        channel_allocation = np.array([False] * self.SCs_of_CPA)  # 每个SC最多一个用户
        user_SC_count = np.zeros(self.num_of_CPA, dtype=int)  # 每用户已分配SC数
        pf_metric = np.zeros((self.num_of_CPA, self.SCs_of_CPA))  # 用户-SC PF 值

        # 初始化每个用户

        for u in range(self.num_of_CPA):
            self.CPA[u].SC_idx = []
            self.CPA[u].partner = []
            self.CPA[u].if_P_NOMA = []
            self.CPA[u].if_S_NOMA = []
            self.avg_rate[u] = self.CPA[u].avg_rate + 1e-6
            self.max_rate[u] = 0

            for s in range(self.SCs_of_CPA):
                gain = self.CPA[u].channel_gain[s].item()
                snr = self.Rb_max * gain / self.noise_power
                if snr > self.SINR_max_CPA:
                    snr = self.SINR_max_CPA
                rate = self.sys_SC_bandwidth * np.log2(1 + snr) * self.slot_time / 8
                avg_rate = self.avg_rate[u]
                pf_metric[u, s] = rate / (avg_rate ** 1.2)
        # print(pf_metric)

        # 主调度循环
        while not np.all(channel_allocation):
            # 每轮更新 pf_metric（或可优化为只更新受影响用户）

            # 找当前最大 PF 值的用户-SC组合
            u, s = np.unravel_index(np.argmax(pf_metric), pf_metric.shape)
            if pf_metric[u, s] == -np.inf:
                break  # 没有可分配资源
            # 检查是否超出最大SC限制
            if (max_SCs_per_user is None) or (user_SC_count[u] < max_SCs_per_user):
                # 分配 SC s 给用户 u
                self.CPA[u].SC_idx.append(s)
                self.CPA[u].partner.append(None)
                self.CPA[u].if_P_NOMA.append(None)
                self.CPA[u].if_S_NOMA.append(None)

                self.Scs_UE[s] = u
                channel_allocation[s] = True
                user_SC_count[u] += 1

                # 更新 avg_rate（使用新加的 SC 的速率加入平均）
                gain = self.CPA[u].channel_gain[s].item()
                snr = self.Rb_max * gain / self.noise_power
                if snr > self.SINR_max_CPA:
                    snr = self.SINR_max_CPA
                rate = self.sys_SC_bandwidth * np.log2(1 + snr) * self.slot_time / 8
                self.avg_rate[u] = (1 - self.beta) * self.avg_rate[u] + self.beta * rate
                self.max_rate[u] += rate

                for a in range(self.SCs_of_CPA):
                    if channel_allocation[a]:
                        pf_metric[u, a] = -np.inf
                    else:
                        gain = self.CPA[u].channel_gain[a].item()
                        snr = self.Rb_max * gain / self.noise_power
                        if snr > self.SINR_max_CPA:
                            snr = self.SINR_max_CPA
                        rate = self.sys_SC_bandwidth * np.log2(1 + snr) * self.slot_time / 8
                        avg_rate = self.avg_rate[u]
                        pf_metric[u, a] = rate / (avg_rate ** 1.2)

            # 子信道s已经被选择
            pf_metric[:, s] = -np.inf

    def RRA_obj_fun(self, CTA, CPA=None, match=None, SC_idx=None, step=1):  # 目的是为了实现对协同控制的资源调度
        if CPA is None or match == 0:  # OMA或者CTA独立传播的效用函数

            CTA_gain = CTA.channel_gain[SC_idx].item()
            CTA.pow = self.sinr_th * self.noise_power / CTA_gain
            if CTA.pow > self.Rb_max:
                return -1e6

            CTA.SINR = self.sinr_th
            V = 1 - (1 + CTA.pow * CTA_gain / self.noise_power) ** (-2)  # 计算信道离散值
            CTA.rate = self.sys_SC_bandwidth * np.log2(
                1 + CTA.SINR) * self.mini_time - np.sqrt(
                V / self.size_of_CTA) * np.sqrt(2) * erfinv(1 - 2 * self.error_rate) / np.log(2)  # 短包传输的公式
            CTA.rate = CTA.rate / 8
            if CTA.rate >= self.size_of_CTA:
                packet_loss = 0
            else:
                packet_loss = 5
            if CPA is None:
                complexity = 0
            else:
                complexity = self.alpha0

            if CPA is not None:
                return self.k2 * packet_loss + self.k3 * CPA.rate_all / step + self.k4 * complexity
            else:
                return self.k2 * packet_loss + 1000 + self.k4 * complexity
        else:
            if match == 1:
                CPA_gain = CPA.channel_gain[SC_idx].item()
                CTA_gain = CTA.channel_gain[SC_idx].item()
                a = CPA.idx_Vehicles
                b = CTA.idx_Vehicles[0]
                c = CTA.idx_Vehicles[1]
                interfence_HCP, interfence_HPC = self.CTA_CPA_CSI[b][SC_idx].item(), self.CPA_CTA_CSI[a][c][
                    SC_idx].item()
                complexity = interfence_HPC / CTA_gain + (min(np.linalg.norm(CPA_gain),
                                                              np.linalg.norm(interfence_HCP)) /
                                                          max(np.linalg.norm(CPA_gain),
                                                              np.linalg.norm(interfence_HCP)))
                # self.windows.complexity_many.append(complexity)
                if complexity > self.complexity_max:
                    return -1e10
                if CTA_gain > interfence_HPC and interfence_HCP > CPA_gain:
                    CPA.pow = min(self.SINR_max_CPA * self.noise_power / CPA_gain,
                                  (self.Rb_max * CTA_gain - self.sinr_th * self.noise_power) / (
                                          CTA_gain + self.sinr_th * interfence_HPC),
                                  (self.Rb_max * interfence_HCP - self.sinr_th * self.noise_power) / (
                                          interfence_HCP + self.sinr_th * CPA_gain))
                    CTA.pow = max(self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain,
                                  self.sinr_th * (self.noise_power + CPA.pow * CPA_gain) / interfence_HCP)
                    # print("1-0", CPA.pow, CTA.pow)
                    CPA.SINR = CPA.pow * CPA_gain / (self.noise_power + CTA.pow * interfence_HCP)
                    if CPA.pow < 0 or CTA.pow < 0 or CPA.SINR < self.SINR_min_CPA:
                        return -1e10
                    interface_CPA = 0
                    interface_CTA = CPA.pow * interfence_HPC
                    # print("1-1", CPA.pow, CTA.pow)
                elif CTA_gain > interfence_HPC and interfence_HCP < CPA_gain:
                    CPA.pow = self.SINR_max_CPA * self.noise_power * (CTA_gain + self.sinr_th * interfence_HCP) / (
                            CPA_gain * CTA_gain - self.sinr_th * interfence_HPC * interfence_HCP * self.SINR_max_CPA)
                    CTA.pow = self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain
                    if CPA.pow + CTA.pow > self.Rb_max or CPA.pow < 0 or CTA.pow < 0:
                        CPA.pow = (self.Rb_max * CTA_gain - self.sinr_th * self.noise_power) / (
                                self.sinr_th * interfence_HPC + CTA_gain)
                        CTA.pow = self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain
                    # print("2-0", CPA.pow, CTA.pow)
                    CPA.SINR = CPA.pow * CPA_gain / (self.noise_power + CTA.pow * interfence_HCP)
                    if CPA.pow < 0 or CTA.pow < 0 or CPA.SINR < self.SINR_min_CPA:
                        return -1e10
                    interface_CPA = CTA.pow * interfence_HCP
                    interface_CTA = CPA.pow * interfence_HPC
                    # print("2-1", CPA.pow, CTA.pow)
                else:
                    return -1e10

                rate_max = self.sys_SC_bandwidth * np.log2(
                    1 + self.Rb_max * CPA_gain / self.noise_power) * self.mini_time
                if rate_max > self.rate_max:
                    rate_max = self.rate_max

                CPA.SINR = CPA.pow * CPA_gain / (
                        self.noise_power + interface_CPA)

                if CPA.SINR > self.SINR_max_CPA:
                    CPA.SINR = self.SINR_max_CPA

                CTA.SINR = CTA.pow * CTA_gain / (
                        self.noise_power + interface_CTA)

                CPA_rate = self.sys_SC_bandwidth * np.log2(1 + CPA.SINR) * self.mini_time / 8
                # print("速率", CPA_rate)
                V = 1 - (1 + CTA.pow * CTA_gain / self.noise_power) ** (-2)
                CTA.rate = self.sys_SC_bandwidth * np.log2(
                    1 + CTA.SINR) * self.mini_time - np.sqrt(
                    V / self.size_of_CTA) * np.sqrt(2) * erfinv(1 - 2 * self.error_rate) / np.log(2)  # 短包传输的公式

                CTA.rate = CTA.rate / 8
                if CTA.rate >= self.size_of_CTA:
                    packet_loss = 0
                else:
                    packet_loss = 5

                return self.k1 * CPA_rate * 8 / rate_max + self.k2 * packet_loss + self.k3 * CPA.rate_all / step + self.k4 * complexity

    def get_cdf(self, values):
        """
        Computes the Cumulative Distribution Function (CDF) of the input vector
        """
        values = np.array(values).flatten()
        n = len(values)
        sorted_val = np.sort(values)
        cumulative_prob = np.arange(1, n + 1) / n
        return sorted_val, cumulative_prob

    # 共享子信道这部分还需要再修改下，直接修改数量的话，子信道的索引会有问题
    def UE_matching_erfen(self, step):  # 二分匹配法
        weight_NOMA = np.zeros((self.num_of_SCs, self.num_of_CTA, 2)) - 1e10  # 边的权重 0 OMA 1PD 2 SD  共享子信道下
        for i in range(self.num_of_CTA):
            for j in range(self.SCs_of_CPA_pro, self.num_of_SCs):
                CPA_idx = self.Scs_UE[j]
                if CPA_idx != 100:
                    CPA = self.CPA[CPA_idx]
                    weight_NOMA[j][i][0] = self.RRA_obj_fun(self.CTA[i], CPA, 0, SC_idx=j, step=step)
                    weight_NOMA[j][i][1] = self.RRA_obj_fun(self.CTA[i], CPA, 1, SC_idx=j, step=step)
                else:
                    weight_NOMA[j][i][0] = self.RRA_obj_fun(self.CTA[i], None, match=0, SC_idx=j, step=step)
        W = weight_NOMA
        W_best = W.max(axis=2)  # 每个 SC–CTA 对的最佳权重，shape: [NS, NU]
        mode_best = W.argmax(axis=2)  # 每个 SC–CTA 对上对应的最佳 match 模式 (0/1/2)，shape: [NS, NU]
        W2 = W_best.T  # shape: [NU, NS]
        cost = -W2
        row_ind, col_ind = linear_sum_assignment(cost)
        total_sum = 0
        for CTA_idx, SC_idx in zip(row_ind, col_ind):
            weight_val = W2[CTA_idx, SC_idx]
            if weight_val < -1e10 + 1e2:
                continue
            total_sum += weight_val
            match = mode_best[SC_idx, CTA_idx]  # 0: OMA, 1: P-NOMA, 2: S-NOMA
            CPA_idx = self.Scs_UE[SC_idx]
            if CPA_idx == 100:
                CPA_idx = SC_idx
            if CPA_idx < self.SCs_of_CPA:
                position = self.CPA[CPA_idx].SC_idx.index(SC_idx)
                self.CPA[CPA_idx].partner[position] = CTA_idx
                self.CTA[CTA_idx].SC_idx = SC_idx
                self.CTA[CTA_idx].partner = CPA_idx
                if match == 1:
                    self.CPA[CPA_idx].if_P_NOMA[position] = 1
                    self.CTA[CTA_idx].if_P_NOMA = 1
                elif match == 2:
                    self.CPA[CPA_idx].if_S_NOMA[position] = 1
                    self.CTA[CTA_idx].if_S_NOMA = 1
            else:
                self.CTA[CTA_idx].SC_idx = SC_idx
                self.CTA[CTA_idx].partner = None

    def RRA(self, CTA=None, CPA=None, SC_idx=None, match=None):  ##整体和RRA_obj_fun 类似，但是此时我们已知分配情况
        if CPA is None and CTA is not None:  # OMA或者CTA独立传播的效用函数
            CTA_gain = CTA.channel_gain[SC_idx].item()
            CTA.pow = self.sinr_th * self.noise_power / CTA_gain
            if CTA.pow > self.Rb_max:
                CTA.pow = self.Rb_max

            # CTA.SINR = self.sinr_th
            CTA.SINR = CTA.pow * CTA_gain / self.noise_power
            V = 1 - (1 + CTA.pow * CTA_gain / self.noise_power) ** (-2)  # 计算信道离散值
            CTA.rate = self.sys_SC_bandwidth * np.log2(
                1 + CTA.SINR) * self.mini_time - np.sqrt(
                V / self.size_of_CTA) * np.sqrt(2) * erfinv(1 - 2 * self.error_rate) / np.log(2)  # 短包传输的公式
            CTA.rate = CTA.rate / 8
            if CTA.rate < 180:
                print(CTA.rate)
            return CTA, 0

        if CTA is None and CPA is not None:
            CPA.pow = self.Rb_max
            CPA_gain = CPA.channel_gain[SC_idx].item()
            CPA.SINR = CPA.pow * CPA_gain / self.noise_power
            if CPA.SINR > self.SINR_max_CPA:
                CPA.SINR = self.SINR_max_CPA
            CPA.rate = self.sys_SC_bandwidth * np.log2(1 + CPA.SINR) * self.mini_time / 8
            return CPA, 0

        elif CTA is not None and CPA is not None:
            complexity = 0
            CTA_gain = CTA.channel_gain[SC_idx].item()
            if match == 0:
                CTA.pow = self.sinr_th * self.noise_power / CTA_gain
                if CTA.pow > self.Rb_max:
                    CTA.pow = self.Rb_max
                # CTA.SINR = self.sinr_th
                CTA.SINR = CTA.pow * CTA_gain / self.noise_power
                V = 1 - (1 + CTA.pow * CTA_gain / self.noise_power) ** (-2)  # 计算信道离散值
                CTA.rate = self.sys_SC_bandwidth * np.log2(
                    1 + CTA.SINR) * self.mini_time - np.sqrt(
                    V / self.size_of_CTA) * np.sqrt(2) * erfinv(1 - 2 * self.error_rate) / np.log(2)  # 短包传输的公式
                CTA.rate = CTA.rate / 8
                complexity = self.alpha0

                CPA.pow = 0
                CPA.SINR = 0
                CPA.rate = 0

            elif match == 1:
                CPA_gain = CPA.channel_gain[SC_idx].item()
                a = CPA.idx_Vehicles
                b = CTA.idx_Vehicles[0]
                c = CTA.idx_Vehicles[1]
                interfence_HCP, interfence_HPC = self.CTA_CPA_CSI[b][SC_idx].item(), self.CPA_CTA_CSI[a][c][
                    SC_idx].item()

                if CTA_gain > interfence_HPC and interfence_HCP > CPA_gain:
                    CPA.pow = min(self.SINR_max_CPA * self.noise_power / CPA_gain,
                                  (self.Rb_max * CTA_gain - self.sinr_th * self.noise_power) / (
                                          CTA_gain + self.sinr_th * interfence_HPC),
                                  (self.Rb_max * interfence_HCP - self.sinr_th * self.noise_power) / (
                                          interfence_HCP + self.sinr_th * CPA_gain))
                    CTA.pow = max(self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain,
                                  self.sinr_th * (self.noise_power + CPA.pow * CPA_gain) / interfence_HCP)
                    interface_CPA = 0
                    interface_CTA = CPA.pow * interfence_HPC

                elif CTA_gain > interfence_HPC and interfence_HCP < CPA_gain:
                    CPA.pow = self.SINR_max_CPA * self.noise_power * (CTA_gain + self.sinr_th * interfence_HCP) / (
                            CPA_gain * CTA_gain - self.sinr_th * interfence_HPC * interfence_HCP * self.SINR_max_CPA)
                    CTA.pow = self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain
                    if CPA.pow + CTA.pow > self.Rb_max or CPA.pow < 0 or CTA.pow < 0:
                        CPA.pow = (self.Rb_max * CTA_gain - self.sinr_th * self.noise_power) / (
                                self.sinr_th * interfence_HPC + CTA_gain)
                        CTA.pow = self.sinr_th * (self.noise_power + CPA.pow * interfence_HPC) / CTA_gain
                    CPA.SINR = CPA.pow * CPA_gain / self.noise_power
                    interface_CPA = CTA.pow * interfence_HCP
                    interface_CTA = CPA.pow * interfence_HPC

                CPA.SINR = CPA.pow * CPA_gain / (self.noise_power + interface_CPA)
                if CPA.SINR > self.SINR_max_CPA:
                    CPA.SINR = self.SINR_max_CPA

                CTA.SINR = CTA.pow * CTA_gain / (
                        self.noise_power + interface_CTA)

                CPA.rate = self.sys_SC_bandwidth * np.log2(1 + CPA.SINR) * self.mini_time / 8
                V = 1 - (1 + CTA.pow * CTA_gain / self.noise_power) ** (-2)
                CTA.rate = self.sys_SC_bandwidth * np.log2(
                    1 + CTA.SINR) * self.mini_time - np.sqrt(
                    V / self.size_of_CTA) * np.sqrt(2) * erfinv(1 - 2 * self.error_rate) / np.log(2)  # 短包传输的公式
                CTA.rate = CTA.rate / 8
                complexity = interfence_HPC / CTA_gain + (min(np.linalg.norm(CPA_gain),
                                                              np.linalg.norm(interfence_HCP)) /
                                                          max(np.linalg.norm(CPA_gain),
                                                              np.linalg.norm(interfence_HCP)))

        return CTA, CPA, complexity

    def calculate_mini_slot(self):
        for i in range(self.num_of_CPA):
            if self.CPA[i].SC_idx:
                rate_mini_slot = 0
                for index, SC_idx in enumerate(self.CPA[i].SC_idx):
                    partner = self.CPA[i].partner[index]
                    if self.CPA[i].if_P_NOMA[index]:
                        self.CTA[partner], self.CPA[i], complexity = self.RRA(self.CTA[partner], self.CPA[i], SC_idx, 1)
                        # print("SINR", self.CPA[i].pow, self.CTA[partner].pow, self.CPA[i].rate,
                        #       self.CPA[i].channel_gain[SC_idx].item(),
                        #       self.CTA[partner].channel_gain[SC_idx].item())
                    elif self.CPA[i].if_S_NOMA[index]:
                        self.CTA[partner], self.CPA[i], complexity = self.RRA(self.CTA[partner], self.CPA[i], SC_idx, 2)
                    elif partner is None:  # 一个RB内 一个用户
                        self.CPA[i], complexity = self.RRA(None, self.CPA[i], SC_idx, 0)
                    else:
                        self.CTA[partner], self.CPA[i], complexity = self.RRA(self.CTA[partner], self.CPA[i], SC_idx, 0)
                    if np.isnan(self.CPA[i].rate):
                        continue
                    else:
                        self.CPA[i].rate_all += self.CPA[i].rate
                        rate_mini_slot += self.CPA[i].rate
                    self.complexity += complexity
                    self.windows.complexity += complexity

                self.CPA[i].avg_rate = (1 - self.beta) * self.CPA[i].avg_rate + self.beta * rate_mini_slot  # 平均速率更新
                self.CPA[i].rate_slot += rate_mini_slot

        CTA_packet_loss = 0
        for i in range(self.num_of_CTA):
            id1 = self.CTA[i].id
            self.act_CC_num[id1] += 1

            if self.CTA[i].SC_idx is not None:
                if self.CTA[i].partner is None:
                    self.CTA[i], a = self.RRA(self.CTA[i], None, self.CTA[i].SC_idx, None)
                if self.CTA[i].rate < self.size_of_CTA:
                    CTA_packet_loss += 1
                    # print("rate丢包")
                    self.pls_CC_num[id1] += 1
                self.windows.SINR_CTA.append(self.CTA[i].SINR)
            else:
                CTA_packet_loss += 1
                # print("控制丢包")
                self.pls_CC_num[id1] += 1
        self.CTA_pls += CTA_packet_loss

    def calculate_long_time(self, step):  # 计算CPA的用户速率，用来求长时间的reward
        packet_loss_all = 0
        delay_all = np.zeros(self.num_of_CPA)
        syn = 1
        for i in range(self.num_of_CPA):
            # print("总传输速率", self.CPA[i].rate_all)
            if self.CPA[i].rate_all < self.size_of_CPA:
                syn = 0
                print("不同步")
                self.windows.CPA_pls += 1
            delay_all[i] = self.size_of_CPA / self.CPA[i].rate_all * 50
            a = self.CPA[i].rate_all * 8 / (self.sys_SC_bandwidth * self.slot_time * 50)
            sinr = np.power(2, a) - 1
            # print("速率", self.CPA[i].rate_all, np.sum(self.windows.num_CTA))
            self.windows.SINR_CPA.append(sinr)

        self.windows.CPA_delay_dist.append(delay_all)
        self.windows.CPA_Synchronicity += syn
        self.windows.CTA_pls_all.append(self.CTA_pls)

        Synchronicity = self.w1 if syn == 1 else 0  # 几个感知数据未同步

        self.windows.qos_CPA = self.w2 * np.exp(-1 * np.mean(self.windows.CPA_delay_dist) / 50) + Synchronicity
        # print(self.windows.qos_CPA,self.w4 * (1 - self.windows.complexity / 1000))

        self.windows.qos_CTA = self.w3 * np.exp(
            -1 * self.windows.CTA_pls_all[0] / np.sum(self.windows.num_CTA))

    def calculate_slot(self, TTI):
        for i in range(self.num_of_CPA):
            if self.CPA[i].SC_idx:
                self.qos_CP += self.w1 * self.CPA[i].rate_slot / self.max_rate[i]
                self.qos_CP2 += self.w1 * self.CPA[i].rate_slot / self.max_rate[i]
            # else:
            #     self.qos_CP += self.w1

        # print("过程", self.qos_CP,self.max_rate[i])
        if np.sum(self.act_CC_num) != 0:
            self.qos_CC += self.w3 * np.exp(-(np.sum(self.pls_CC_num) / np.sum(self.act_CC_num)))
        else:
            self.qos_CC += self.w3

        if TTI == self.max_slot:
            self.done = 1
            self.qos_CP += self.w2 * self.windows.CPA_Synchronicity / 100  # 100个感知周期的同步率
            self.qos_CC += self.w4 * np.exp(-(np.sum(self.windows.CTA_pls_all) / np.sum(self.windows.num_CTA)))

            print("同步率", self.windows.CPA_Synchronicity / 100)
            print("总丢包", np.sum(self.windows.CTA_pls_all) / np.sum(self.windows.num_CTA))

        # if self.max < self.complexity:
        #     self.max = self.complexity
        #     print(self.max)
        self.qos_complexity += self.complexity / 4 * self.w5

        self.reward = self.qos_CP2 + self.qos_CC - self.qos_complexity

    def getState_agent(self):

        state = np.array(self.qos_CP2)
        state = np.hstack((state, self.complexity))
        state = np.hstack((state, 0.5 - self.windows.complexity / 0.5))
        state = np.hstack((state, self.CTA_pls / 5 - 0 / 5))
        state = np.hstack((state, np.sum(self.windows.num_CTA) / 1000 - 0.5))
        state = np.hstack((state, self.au / 0.25 - 0.5))
        state = np.hstack((state, np.array(self.avg_pathloss_CP) / 1e-7 - 0.5))
        state = np.hstack((state, np.array(self.avg_pathloss_CC) / 1e-7 - 0.5))  # pathloss
        state = np.hstack((state, np.array(self.CPA_CTA_pathloss).flatten() / 1e-7 - 0.5))  # pathloss
        state = np.hstack((state, np.array(self.CTA_CPA_pathloss) / 1e-7 - 0.5))  # pathloss
        return state

    def reset(self):
        # np.random.seed(42)
        self.windows = None
        self.windows = Slidingwindow(self.length)  # 滑动窗
        self.windows.start_time = 11000  # 从11s处开始
        # 初始化参数，主要是为了第一个state

        #
        self.windows.avg_gain_CP = np.zeros((self.CP_num, self.num_of_SCs))
        self.CTA_CSI = np.zeros((self.CC_num - 2, self.num_of_SCs))
        self.CPA_CTA_CSI = np.zeros((self.CP_num, self.CC_num, self.num_of_SCs))
        self.CTA_CPA_CSI = np.zeros((self.CC_num, self.num_of_SCs))
        #
        self.au_all = 0
        self.CPA = None
        self.CTA_pls = 0
        self.complexity = 0
        self.CP_cav, self.CC_cav = self.get_vehicle_position()
        self.get_CTA_all()
        self.get_user_pathloss()
        self.CTA_traffic_Pattern(0)
        self.done = 0

        self.windows.qos_CPA = 0
        self.windows.qos_CTA = 0
        self.windows.num_CTA.append(0)
        self.windows.CPA_delay_dist.append(0)
        self.windows.CTA_pls_all.append(0)
        self.windows.complexity = 0
        self.qos_CP = 0
        self.qos_CP2 = 0
        self.qos_CC = 0
        self.qos_complexity = 0
        self.get_avg_pathloss()

    def step_mini_slot(self, step):  # 计算了所有用户和RB会产生的reward
        self.CTA_traffic()
        self.UE_matching_erfen(step)
        self.calculate_mini_slot()
        for i in range(self.num_of_CPA):
            if self.CPA[i].SC_idx:
                for index, item in enumerate(self.CPA[i].SC_idx):
                    self.CPA[i].if_P_NOMA[index] = 0
                    self.CPA[i].if_S_NOMA[index] = 0
                    self.CPA[i].partner[index] = None

    def step_long_time(self, step):
        self.windows.start_time += 50
        self.Scs_UE = [100 for _ in range(self.num_of_SCs)]  # 信道归一化
        self.CP_cav, self.CC_cav = None, None
        self.CP_cav, self.CC_cav = self.get_vehicle_position()
        self.CTA_all = None
        self.CTA_all = [CTAs_all() for i in range(self.CC_num - 2)]  # 修改
        self.CPA = None
        self.get_CTA_all()
        self.CTA_traffic_Pattern(step)
        self.get_user_pathloss()  # 计算所有的pathloss
        self.CTA_pls = 0
        self.windows.CPA_pls = 0
        self.windows.complexity = 0
        self.get_avg_pathloss()

    def step_20ms(self):
        self.qos_CP = 0
        self.qos_CP2 = 0
        self.qos_CC = 0
        self.qos_complexity = 0
    def step_slot(self):

        self.user_CSI()
        # self.get_avg_gain()
        self.schedule_CPA()
        self.complexity = 0
