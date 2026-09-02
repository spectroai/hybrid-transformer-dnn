import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ast

# 与 Train_model_Transformer_DNN.ipynb 训练脚本完全一致的序列长度
MAX_LEN = 300


class SpectrumTransformer(nn.Module):
    """与训练脚本 Train_model_Transformer_DNN.ipynb 一致的 Transformer 结构。"""
    def __init__(self, d_model=256, nhead=4, num_layers=4, max_seq_len=MAX_LEN,
                 num_precursor_types=79, dropout=0.1):
        super(SpectrumTransformer, self).__init__()
        self.peak_embed = nn.Sequential(
            nn.Linear(2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.precursor_embedding = nn.Embedding(num_precursor_types, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

    def forward(self, mz_values, intensities, precursor_types):
        # 与训练一致：padding 位置为 intensity==0
        padding_mask = (intensities == 0)  # [B, L], True=ignore

        peaks = torch.stack([mz_values, intensities], dim=-1)  # [B, L, 2]
        x = self.peak_embed(peaks)  # [B, L, d]

        prec_emb = self.precursor_embedding(precursor_types)  # [B, d]
        x = x + prec_emb.unsqueeze(1)

        L = mz_values.shape[1]
        x = x + self.pos_embedding[:L].unsqueeze(0)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 仅对非 padding 位置做 mean（与训练一致）
        valid = (~padding_mask).unsqueeze(-1).float()
        x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        return self.output_projection(x)


# ===============================
# 2️⃣ 混合模型定义 - 与训练脚本一致
# ===============================
class MultiLabelDNN(nn.Module):
    """与 Train_model_Transformer_DNN.ipynb 的 MultiLabelDNN 一致。"""
    def __init__(self, input_size, output_size, dropout_rate=0.1,
                 transformer_dim=256, num_precursor_types=79):
        super(MultiLabelDNN, self).__init__()
        self.transformer = SpectrumTransformer(
            d_model=transformer_dim,
            num_precursor_types=num_precursor_types,
            dropout=dropout_rate
        )
        fusion_in = input_size + transformer_dim

        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mz_values=None, intensities=None, precursor_types=None):
        if mz_values is not None and intensities is not None and precursor_types is not None:
            tf = self.transformer(mz_values, intensities, precursor_types)
            fused = torch.cat([x, tf], dim=1)
            return self.fusion_network(fused)
        return self.fusion_network(x)

# ===============================
# 3️⃣ Transformer DNN指纹生成器 - 用于run_efficient_similarity.py
# ===============================
class TransformerDNNFingerprintGenerator:
    """Transformer DNN指纹生成器 - 用于run_efficient_similarity.py
    支持不区分正负模式的统一模型：从checkpoint加载precursor_to_idx以正确编码离子类型。
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', ion_mode: str = 'negative'):
        """
        初始化Transformer DNN指纹生成器
        
        Args:
            model_path: 训练好的Transformer DNN模型路径（如 DNN_model/Transformer_DNN/checkpoints/best_model_complete.pth）
            device: 计算设备 ('cpu' 或 'cuda')
            ion_mode: 离子模式 ('positive' 或 'negative')，仅当checkpoint中无precursor_to_idx时用作后备
        """
        self.device = device
        self.model_path = model_path
        self.ion_mode = ion_mode
        
        # 设置bin参数 - 与训练模型完全一致
        self.mz_min, self.mz_max, self.bin_size = 20, 1200, 1
        self.n_bins = self.mz_max - self.mz_min  # 1180 bins
        self.bin_edges = np.arange(self.mz_min, self.mz_max + self.bin_size, self.bin_size)
        
        # 先加载 checkpoint，若包含 precursor_to_idx 则使用（统一模型）；否则按 ion_mode 设置
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'precursor_to_idx' in checkpoint:
            self.precursor_to_idx = dict(checkpoint['precursor_to_idx'])
            # 按索引排序得到与训练时一致的顺序
            self.unique_precursors = [k for k, _ in sorted(self.precursor_to_idx.items(), key=lambda x: x[1])]
            print(f"[INFO] 使用checkpoint中的前体类型映射（统一模型，不区分正负），共 {len(self.unique_precursors)} 种")
        else:
            # 后备：根据离子模式设置前体类型映射
            if ion_mode.lower() == 'negative':
                self.precursor_to_idx = {
                    '[M-H]-': 0, 'M-H': 1, '[M+FA-H]-': 2, '[M+K-2H]-': 3, 'M+NH4': 4, 
                    '[M+Cl]-': 5, '[M-H20-H]-': 6, '[M+HCOO]-': 7, '[2M-2H+Na]-': 8, '[2M-H]-': 9, 
                    '[M+CH3COO]-': 10, 'M+Cl': 11, '[M-2H+Na]-': 12, 'M+H': 13, 'carotenoid': 14, 
                    'M+K': 15, '[M+HAc-H]-': 16, '[2M+FA-H]-': 17, '[M+HOO]-': 18, '[M-H]': 19, 
                    '[M+Na-2H]-': 20, 'M+Na': 21, '[M-H]1-': 22, '[M+H]+': 23, '[M-2H]-': 24, 
                    '[M+COOH]-': 25, '[M-CH3]-': 26, '[M-C3H7O2]-': 27, '[M+CH3COOH-H]-': 28, '[M-C2H3O]-': 29, 
                    '[M-CO2-H]-': 30, '[M]-': 31, '[M+H]-': 32, '[M-H2O+H]-': 33, '[M-2H]--': 34, 
                    '[M-H-CO2-2HF]-': 35, '[M+Hac+Na-2H]-': 36, '[M+Hac-H]-': 37, '[M+Na]+': 38
                }
            else:
                self.precursor_to_idx = {
                    '[M+H]+': 0, '[M+K]+': 1, '[M+NH4]+': 2, '[M+Na]+': 3, '[M-H2O+H]+': 4, 
                    'M-H': 5, '[M+H]': 6, '[2M+Na]+': 7, '[M-2H2O+H]+': 8, '[2M+H]+': 9, 
                    'M+Na': 10, '[2M+NH4]+': 11, 'M+NH4': 12, 'M+K': 13, '[M]+*': 14, 
                    'carotenoid': 15, 'M+H': 16, '[M+H-H2O]-': 17, '[2M+K]+': 18, '[2M+Ca]2+': 19, 
                    '[M+H-H2O]': 20, '[M+ACN+H]+': 21, '[3M+Ca]2+': 22, '[M-CH3]+': 23, '[M]+': 24, 
                    '[M-H+2Na]+': 25, '[2M-2H2O+H]+': 26, '[2M-H2O+H]+': 27, '[M+H+CH3CN]+': 28, '[4M+Ca]2+': 29, 
                    '[M+Ca]2+': 30, '[M+H-2H2O]+': 31, '[M-3H2O+H]+': 32, '[M-4H2O+H]+': 33, '[M-5H2O+H]+': 34, 
                    '[3M+Na]+': 35, '[M-H]-': 36, '[M+Na+CH3CN]+': 37, '[M-OH]+': 38, '[M-H+Li]+*': 39, 
                    '[M-H+Na]+*': 40, '[5M+Ca]2+': 41, '[M+H-H2O]+': 42, '[M+2H]++': 43, '[M-H]+': 44, 
                    '[M+H+Na]2+': 45, '[M-2H2O+2H]2+': 46, '[M-3H2O+2H]2+': 47, '[2M-H+2Na]+': 48, '[M+2H]+': 49, 
                    '[M+CH3OH+H]+': 50, '[M+CH3]+': 51, 'carotenoids': 52, '[M+Li]+*': 53, '[M+Na]+*': 54, 
                    '[M+2H]2+': 55, '[M+ACN+NH4]+': 56, 'M+': 57, '[3M+Ca-H]+': 58, '[M-MeOH+H]+': 59, 
                    '[3M+K]+': 60, '[M+FA+H]+': 61, 'M+2Na': 62, '[M+H-99]+': 63, '[3M+NH4]+': 64, 
                    '[M-H2O+H]': 65, '[2M-3H2O+H]+': 66, '[M-2H2O+NH4]+': 67, '[M+H-C12H20O9]+': 68, '[M]++': 69, 
                    '[M+2H-NH4]+': 70, '[M-NH4+2H]+': 71, '[M+H-NH3]+': 72, '[M+H-C9H10O5]+': 73, '[2M+H+CH3CN]+': 74, 
                    '[M+H-SO3]+': 75, '[M-SO3+H]+': 76, '[M-C6H10O5+H]+': 77, '[M+]': 78
                }
            self.unique_precursors = list(self.precursor_to_idx.keys())
            print(f"[INFO] checkpoint无precursor_to_idx，使用离子模式 '{ion_mode}' 的默认映射")
        
        # 加载模型（依赖 self.unique_precursors 已设置）
        self.model = self._load_model(checkpoint)
        
        print(f"Transformer DNN指纹生成器初始化完成")
        print(f"模型路径: {model_path}")
        print(f"设备: {device}")
        print(f"输入维度: {self.n_bins}")
        print(f"前体类型数量: {len(self.unique_precursors)}")
    
    def _load_model(self, checkpoint: dict) -> nn.Module:
        """加载训练好的 Transformer DNN 模型（与 Train_model_Transformer_DNN.ipynb 结构一致）。"""
        try:
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"[INFO] 从checkpoint中加载model_state_dict")
            else:
                state_dict = checkpoint
                print(f"[INFO] 直接加载checkpoint")
            
            num_prec = len(self.unique_precursors)
            # 与训练脚本一致：dropout_rate=0.1，无 use_transformer
            model = MultiLabelDNN(
                input_size=self.n_bins,
                output_size=881,
                dropout_rate=0.1,
                transformer_dim=256,
                num_precursor_types=num_prec
            )
            # 兼容旧 checkpoint 中 pos_encoding 键名（训练脚本为 pos_embedding）
            state_dict = dict(state_dict)
            if 'transformer.pos_encoding' in state_dict and 'transformer.pos_embedding' not in state_dict:
                state_dict['transformer.pos_embedding'] = state_dict.pop('transformer.pos_encoding')
            model.load_state_dict(state_dict, strict=True)
            
            model.eval()
            model.to(self.device)
            print(f"[SUCCESS] Transformer DNN 模型加载成功 (num_precursor_types={num_prec}, seq_len={MAX_LEN})")
            return model
            
        except Exception as e:
            print(f"[ERROR] Transformer DNN模型加载失败: {e}")
            import traceback
            print(f"[ERROR] 详细错误信息: {traceback.format_exc()}")
            raise
    
    def spectrum_to_vector(self, mz_values, intensities):
        """将光谱数据转换为向量（与训练时相同的处理）"""
        # 安全处理缺失值
        if mz_values is None or (isinstance(mz_values, (list, np.ndarray)) and len(mz_values) == 0):
            mz_values = []
        if intensities is None or (isinstance(intensities, (list, np.ndarray)) and len(intensities) == 0):
            intensities = []

        # 字符串解析
        if isinstance(mz_values, str):
            mz_values = ast.literal_eval(mz_values)
        if isinstance(intensities, str):
            intensities = ast.literal_eval(intensities)
        
        mz_values = np.array(mz_values)
        intensities = np.array(intensities)

        # 只取范围内的峰
        mask = (mz_values >= self.mz_min) & (mz_values <= self.mz_max)
        mz_values = mz_values[mask]
        intensities = intensities[mask]

        # 生成 binned vector
        spectrum_vec, _ = np.histogram(
            mz_values, bins=self.bin_edges, weights=intensities
        )

        # 转为 float32
        spectrum_vec = spectrum_vec.astype(np.float32)

        # 归一化：除以最大峰值
        max_val = spectrum_vec.max()
        if max_val > 0:
            spectrum_vec = spectrum_vec / max_val

        return spectrum_vec
    
    def prepare_transformer_input(self, mz_values, intensities, max_len=MAX_LEN):
        """准备 Transformer 输入，与训练脚本一致：先按最大强度归一化，再 top-k 或 pad 到 max_len。"""
        if mz_values is None or (isinstance(mz_values, (list, np.ndarray)) and len(mz_values) == 0):
            mz_values = []
        if intensities is None or (isinstance(intensities, (list, np.ndarray)) and len(intensities) == 0):
            intensities = []

        if isinstance(mz_values, str):
            mz_values = ast.literal_eval(mz_values)
        if isinstance(intensities, str):
            intensities = ast.literal_eval(intensities)

        mz_values = np.array(mz_values, dtype=np.float32)
        intensities = np.array(intensities, dtype=np.float32)

        mask = (mz_values >= self.mz_min) & (mz_values <= self.mz_max)
        mz_values = mz_values[mask]
        intensities = intensities[mask]

        # 与训练一致：先按谱图内最大强度归一化
        mx = float(intensities.max()) if intensities.size > 0 else 1.0
        if mx == 0:
            mx = 1.0
        intensities = intensities / mx

        L = len(mz_values)
        if L > max_len:
            top = np.argsort(intensities)[-max_len:]
            mz_values = mz_values[top]
            intensities = intensities[top]
        elif L < max_len:
            pad_len = max_len - L
            mz_values = np.pad(mz_values, (0, pad_len), 'constant', constant_values=0)
            intensities = np.pad(intensities, (0, pad_len), 'constant', constant_values=0)

        return mz_values.astype(np.float32), intensities.astype(np.float32)
    
    def generate_fingerprint(self, mz_values, intensities, precursor_type, threshold=0.3, binary=True):
        """
        生成指纹向量
        
        Args:
            mz_values: m/z值列表
            intensities: 强度值列表
            precursor_type: 前体类型
            threshold: 分类阈值
            binary: 是否返回二进制指纹
            
        Returns:
            fingerprint: 指纹向量
        """
        try:
            # 数据预处理
            spectrum_vector = self.spectrum_to_vector(mz_values, intensities)
            mz_tensor, intensity_tensor = self.prepare_transformer_input(mz_values, intensities)
            
            # 前体类型编码
            precursor_idx = self.precursor_to_idx.get(precursor_type, 0)
            
            # 转换为tensor
            spectrum_tensor = torch.from_numpy(spectrum_vector).unsqueeze(0).to(self.device)
            mz_tensor = torch.from_numpy(mz_tensor).unsqueeze(0).to(self.device)
            intensity_tensor = torch.from_numpy(intensity_tensor).unsqueeze(0).to(self.device)
            precursor_tensor = torch.tensor([precursor_idx], dtype=torch.long).to(self.device)
            
            # 推理
            with torch.no_grad():
                # 注意：参数顺序必须与MultiLabelDNN.forward()方法一致
                # forward(self, x, mz_values=None, intensities=None, precursor_types=None)
                outputs = self.model(spectrum_tensor, mz_tensor, intensity_tensor, precursor_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                
                # 确保probabilities是一维数组
                if probabilities.ndim > 1:
                    probabilities = probabilities.flatten()
                
                if binary:
                    fingerprint = (probabilities > threshold).astype(int)
                else:
                    fingerprint = probabilities
            
            return fingerprint
            
        except Exception as e:
            print(f"指纹生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None


# ===============================
# 4️⃣ 光谱指纹预测器 - 用于独立使用
# ===============================
class SpectrumPredictor:
    """
    光谱指纹预测器 - 加载最优模型并进行推理
    """
    def __init__(self, model_path, bin_params=None, device=None):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型路径
            bin_params: 分bin参数，与训练时一致
            device: 推理设备
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 设置bin参数（与训练时一致）
        if bin_params is None:
            self.mz_min, self.mz_max, self.bin_size = 20, 1200, 1
        else:
            self.mz_min, self.mz_max, self.bin_size = bin_params
        
        self.bin_edges = np.arange(self.mz_min, self.mz_max + self.bin_size, self.bin_size)
        
        # 模型和参数
        self.model = None
        self.model_params = None
        self.preprocessor = None
        self.precursor_to_idx = None
        self.unique_precursors = None
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型和参数"""
        try:
            # 加载保存的模型状态
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取模型参数
            input_size = 1180  # 与训练时一致
            output_size = 881  # 与训练时一致
            transformer_dim = 256
            num_precursor_types = 39  # 根据实际调整
            
            # 创建模型结构
            self.model = MultiLabelDNN(
                input_size=input_size,
                output_size=output_size,
                dropout_rate=0.2,  # 推理时dropout不起作用
                use_transformer=True,
                transformer_dim=transformer_dim,
                num_precursor_types=num_precursor_types
            ).to(self.device)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()  # 设置为评估模式
            
            # 保存模型参数
            self.model_params = checkpoint.get('params', {})
            
            print(f"✅ 模型加载成功 from {self.model_path}")
            print(f"✅ 设备: {self.device}")
            print(f"✅ 模型参数: {self.model_params}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def set_precursor_mapping(self, precursor_to_idx, unique_precursors):
        """
        设置前体类型映射（必须与训练时一致）
        
        Args:
            precursor_to_idx: 前体类型到索引的映射字典
            unique_precursors: 唯一前体类型列表
        """
        self.precursor_to_idx = precursor_to_idx
        self.unique_precursors = unique_precursors
        print(f"✅ 前体类型映射设置完成，共{len(unique_precursors)}种前体类型")
    
    def spectrum_to_vector(self, mz_values, intensities):
        """
        将光谱数据转换为向量（与训练时相同的处理）
        """
        # 安全处理缺失值
        if mz_values is None or (isinstance(mz_values, (list, np.ndarray)) and len(mz_values) == 0):
            mz_values = []
        if intensities is None or (isinstance(intensities, (list, np.ndarray)) and len(intensities) == 0):
            intensities = []

        # 字符串解析
        if isinstance(mz_values, str):
            mz_values = ast.literal_eval(mz_values)
        if isinstance(intensities, str):
            intensities = ast.literal_eval(intensities)
        
        mz_values = np.array(mz_values)
        intensities = np.array(intensities)

        # 只取范围内的峰
        mask = (mz_values >= self.mz_min) & (mz_values <= self.mz_max)
        mz_values = mz_values[mask]
        intensities = intensities[mask]

        # 生成 binned vector
        spectrum_vec, _ = np.histogram(
            mz_values, bins=self.bin_edges, weights=intensities
        )

        # 转为 float32
        spectrum_vec = spectrum_vec.astype(np.float32)

        # 归一化：除以最大峰值
        max_val = spectrum_vec.max()
        if max_val > 0:
            spectrum_vec = spectrum_vec / max_val

        return spectrum_vec
    
    def prepare_transformer_input(self, mz_values, intensities, max_len=1000):
        """准备Transformer输入数据（与训练时相同）"""
        # 安全处理缺失值
        if mz_values is None or (isinstance(mz_values, (list, np.ndarray)) and len(mz_values) == 0):
            mz_values = []
        if intensities is None or (isinstance(intensities, (list, np.ndarray)) and len(intensities) == 0):
            intensities = []

        # 字符串解析
        if isinstance(mz_values, str):
            mz_values = ast.literal_eval(mz_values)
        if isinstance(intensities, str):
            intensities = ast.literal_eval(intensities)
        
        mz_values = np.array(mz_values)
        intensities = np.array(intensities)

        # 只取范围内的峰
        mask = (mz_values >= self.mz_min) & (mz_values <= self.mz_max)
        mz_values = mz_values[mask]
        intensities = intensities[mask]
        
        # 截断或填充到固定长度
        if len(mz_values) > max_len:
            # 选择强度最高的峰
            top_indices = np.argsort(intensities)[-max_len:]
            mz_values = mz_values[top_indices]
            intensities = intensities[top_indices]
        elif len(mz_values) < max_len:
            # 填充零
            pad_len = max_len - len(mz_values)
            mz_values = np.pad(mz_values, (0, pad_len), 'constant', constant_values=0)
            intensities = np.pad(intensities, (0, pad_len), 'constant', constant_values=0)
        
        return mz_values.astype(np.float32), intensities.astype(np.float32)
    
    def predict_single(self, mz_values, intensities, precursor_type, threshold=0.5):
        """
        预测单个样本的指纹
        
        Args:
            mz_values: m/z值列表
            intensities: 强度值列表
            precursor_type: 前体类型
            threshold: 分类阈值
            
        Returns:
            predictions: 预测的指纹向量
            probabilities: 预测概率
        """
        if self.precursor_to_idx is None:
            raise ValueError("请先设置前体类型映射 using set_precursor_mapping()")
        
        # 数据预处理
        spectrum_vector = self.spectrum_to_vector(mz_values, intensities)
        mz_tensor, intensity_tensor = self.prepare_transformer_input(mz_values, intensities)
        
        # 前体类型编码
        precursor_idx = self.precursor_to_idx.get(precursor_type, 0)
        
        # 转换为tensor
        spectrum_tensor = torch.from_numpy(spectrum_vector).unsqueeze(0).to(self.device)
        mz_tensor = torch.from_numpy(mz_tensor).unsqueeze(0).to(self.device)
        intensity_tensor = torch.from_numpy(intensity_tensor).unsqueeze(0).to(self.device)
        precursor_tensor = torch.tensor([precursor_idx], dtype=torch.long).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(spectrum_tensor, mz_tensor, intensity_tensor, precursor_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            
            # 确保probabilities是一维数组
            if probabilities.ndim > 1:
                probabilities = probabilities.flatten()
            
            predictions = (probabilities > threshold).astype(int)
        
        return predictions, probabilities
    
    def predict_batch(self, data_list, threshold=0.5, batch_size=32):
        """
        批量预测
        
        Args:
            data_list: 数据列表，每个元素为(mz_values, intensities, precursor_type)
            threshold: 分类阈值
            batch_size: 批大小
            
        Returns:
            all_predictions: 所有预测结果
            all_probabilities: 所有预测概率
        """
        if self.precursor_to_idx is None:
            raise ValueError("请先设置前体类型映射 using set_precursor_mapping()")
        
        all_predictions = []
        all_probabilities = []
        
        # 分批处理
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i+batch_size]
            
            batch_spectrum = []
            batch_mz = []
            batch_intensity = []
            batch_precursor = []
            
            # 预处理批次数据
            for mz_values, intensities, precursor_type in batch_data:
                spectrum_vector = self.spectrum_to_vector(mz_values, intensities)
                mz_tensor, intensity_tensor = self.prepare_transformer_input(mz_values, intensities)
                precursor_idx = self.precursor_to_idx.get(precursor_type, 0)
                
                batch_spectrum.append(spectrum_vector)
                batch_mz.append(mz_tensor)
                batch_intensity.append(intensity_tensor)
                batch_precursor.append(precursor_idx)
            
            # 转换为tensor
            spectrum_tensor = torch.from_numpy(np.array(batch_spectrum)).to(self.device)
            mz_tensor = torch.from_numpy(np.array(batch_mz)).to(self.device)
            intensity_tensor = torch.from_numpy(np.array(batch_intensity)).to(self.device)
            precursor_tensor = torch.tensor(batch_precursor, dtype=torch.long).to(self.device)
            
            # 批次推理
            with torch.no_grad():
                outputs = self.model(spectrum_tensor, mz_tensor, intensity_tensor, precursor_tensor)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                predictions = (probabilities > threshold).astype(int)
            
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def predict_from_dataframe(self, df, mz_col='mz_values', intensity_col='intensities', 
                             precursor_col='precursor_type', threshold=0.5, batch_size=32):
        """
        从DataFrame批量预测
        
        Args:
            df: 包含光谱数据的DataFrame
            mz_col: m/z列名
            intensity_col: 强度列名
            precursor_col: 前体类型列名
            threshold: 分类阈值
            batch_size: 批大小
            
        Returns:
            result_df: 包含预测结果的DataFrame
        """
        # 准备数据
        data_list = []
        for _, row in df.iterrows():
            data_list.append((
                row[mz_col],
                row[intensity_col],
                row[precursor_col]
            ))
        
        # 批量预测
        predictions, probabilities = self.predict_batch(data_list, threshold, batch_size)
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['fingerprint_predictions'] = list(predictions)
        result_df['fingerprint_probabilities'] = list(probabilities)
        
        return result_df
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'device': str(self.device),
            'model_path': self.model_path,
            'input_size': 1180,
            'output_size': 881,
            'bin_params': (self.mz_min, self.mz_max, self.bin_size),
            'precursor_types_count': len(self.unique_precursors) if self.unique_precursors else 0
        }
        return info
