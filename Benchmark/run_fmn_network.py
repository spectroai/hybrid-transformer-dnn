#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流式融合分子网络构建脚本（适用于大数据量）

本文件提供以下功能：
1. 流式数据加载（分批加载，避免内存溢出）
2. 流式Cosine相似度计算（不存储完整矩阵）
3. 流式Transformer_DNN相似度计算（从CSV读取嵌入向量）
4. 增量融合网络构建（边计算边添加，只保留满足条件的边）

核心特性：
- 适用于大数据量（9万组数据）
- 内存优化：不存储完整相似度矩阵
- 融合两种算法：Cosine + Transformer_DNN

使用方法：
    直接运行：python integrated_similarity_and_network.py
    （在文件末尾的if __name__ == "__main__"部分修改参数）
    
    或作为模块导入：
    from integrated_similarity_and_network import main_streaming
    main_streaming(csv_file="your_file.csv", embedding_file="your_embeddings.csv", ...)
"""

import numpy as np
import pandas as pd
import ast
import argparse
import warnings
import os
import re
from typing import List, Dict, Optional, Tuple
import networkx as nx
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore')


# ============================================================================
# 0. 辅助函数：解析离子模式
# ============================================================================

def parse_ion_mode(precursor_type):
    """
    解析离子模式
    返回: 'positive', 'negative', 或 'unknown'
    """
    if pd.isna(precursor_type):
        return 'unknown'
    
    precursor_type = str(precursor_type).strip()
    
    # 正离子模式: [M+H]+, [M+Na]+, [M+K]+, [M+NH4]+, 等
    if re.search(r'\[M\+[^-\]]+\]\+', precursor_type):
        return 'positive'
    
    # 负离子模式: [M-H]-, [M+HCOOH-H]-, [M+Cl]-, [M+CH3OH-H]-, 等
    if re.search(r'\[M[^+\]]+\]-', precursor_type):
        return 'negative'
    
    return 'unknown'


# ============================================================================
# 1. 流式数据加载器
# ============================================================================

class StreamingDataLoader:
    """流式数据加载器，支持分批加载大数据文件"""
    
    def __init__(self, csv_file_path: str, batch_size: int = 5000, max_samples: Optional[int] = None):
        """
        初始化流式数据加载器
        
        参数:
            csv_file_path: CSV文件路径
            batch_size: 每批加载的数据量
            max_samples: 最大处理样本数（用于测试）
        """
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.max_samples = max_samples
        
        # 获取总数据量
        self.total_count = self._count_rows()
        
        # 如果指定了max_samples，限制总数
        if self.max_samples is not None:
            self.total_count = min(self.total_count, self.max_samples)
        
        print(f"流式数据加载器初始化完成")
        print(f"  数据文件: {csv_file_path}")
        print(f"  总数据量: {self.total_count}")
        print(f"  批次大小: {batch_size}")
    
    def _count_rows(self) -> int:
        """统计CSV文件的行数"""
        with open(self.csv_file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # 减去表头
    
    def get_total_count(self) -> int:
        """获取总数据量"""
        return self.total_count
    
    def load_batch(self, start_idx: int, end_idx: int) -> Tuple[List[Dict], List[float]]:
        """
        加载指定范围的数据批次
        
        参数:
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
        
        返回:
            batch_spectra: 谱图数据列表
            batch_precursor_mz: 前体质量列表
        """
        # 限制end_idx不超过total_count
        end_idx = min(end_idx, self.total_count)
        
        # 读取指定行的数据
        # 注意：gnps_clean_testdata_originindex.csv 的第一列是索引（行名）
        # 使用 index_col=0 让pandas将第一列作为索引，但我们需要确保行号对应正确
        # skiprows=range(1, start_idx + 1) 会跳过第1行（表头）到第start_idx行
        # 然后读取从第start_idx+1行开始的nrows行数据
        df = pd.read_csv(self.csv_file_path, skiprows=range(1, start_idx + 1), 
                        nrows=end_idx - start_idx, index_col=0)
        
        batch_spectra = []
        batch_precursor_mz = []
        
        for idx, row in df.iterrows():
            try:
                # 解析m/z和intensity数组（支持多种列名）
                mz_str = row.get('mz_values') or row.get('mz', '[]')
                intensity_str = row.get('intensities') or row.get('intensity', '[]')
                
                # 安全解析
                if isinstance(mz_str, str):
                    try:
                        mz_array = np.array(ast.literal_eval(mz_str))
                    except:
                        mz_array = np.array([])
                elif isinstance(mz_str, (list, np.ndarray)):
                    mz_array = np.array(mz_str)
                else:
                    mz_array = np.array([])
                
                if isinstance(intensity_str, str):
                    try:
                        intensity_array = np.array(ast.literal_eval(intensity_str))
                    except:
                        intensity_array = np.array([])
                elif isinstance(intensity_str, (list, np.ndarray)):
                    intensity_array = np.array(intensity_str)
                else:
                    intensity_array = np.array([])
                
                # 获取其他信息
                smiles = row.get('smiles', '')
                # 尝试多种方式获取precursor_mz
                precursor_mz = 0.0
                if 'precursor_mz' in row:
                    precursor_mz = float(row.get('precursor_mz', 0.0))
                elif 'precursor_type' in row:
                    # 如果只有precursor_type，尝试从mz_values的最大值估算
                    if len(mz_array) > 0:
                        precursor_mz = float(np.max(mz_array))
                
                # 获取并解析离子模式
                precursor_type = row.get('precursor_type', '')
                ion_mode = parse_ion_mode(precursor_type)
                
                # 获取cluster_id
                cluster_id = row.get('cluster_id', '')
                if pd.isna(cluster_id):
                    cluster_id = ''
                else:
                    cluster_id = str(cluster_id)
                
                spectrum = {
                    'm/z array': mz_array,
                    'intensity array': intensity_array,
                    'smiles': smiles,
                    'PEPMASS': precursor_mz,
                    'original_index': start_idx + len(batch_spectra),
                    'ion_mode': ion_mode,
                    'precursor_type': precursor_type,
                    'cluster_id': cluster_id
                }
                
                batch_spectra.append(spectrum)
                batch_precursor_mz.append(precursor_mz)
                
            except Exception as e:
                print(f"警告: 解析第 {start_idx + len(batch_spectra)} 行数据时出错: {e}")
                continue
        
        return batch_spectra, batch_precursor_mz


# ============================================================================
# 2. 流式Cosine相似度计算器
# ============================================================================

class StreamingCosineSimilarityCalculator:
    """流式余弦相似度计算器（不存储完整矩阵）"""
    
    def __init__(self, mz_bins: Optional[np.ndarray] = None):
        """
        初始化流式余弦相似度计算器
        
        参数:
            mz_bins: m/z bins数组，如果为None则自动计算
        """
        self.mz_bins = mz_bins
    
    def compute_mz_bins(self, all_spectra: List[Dict]) -> np.ndarray:
        """从所有谱图中计算m/z bins"""
        # 收集所有有效的m/z值
        all_mz_list = []
        for s in all_spectra:
            mz_array = s.get("m/z array", np.array([]))
            if len(mz_array) > 0:
                all_mz_list.append(mz_array)
        
        # 检查是否有有效的m/z数据
        if len(all_mz_list) == 0:
            print("警告: 没有找到有效的m/z数据，使用默认范围 [0, 2000]")
            return np.arange(0, 2000.1, 0.1)
        
        # 合并所有m/z值
        all_mz = np.concatenate(all_mz_list)
        
        if len(all_mz) == 0:
            print("警告: m/z数组为空，使用默认范围 [0, 2000]")
            return np.arange(0, 2000.1, 0.1)
        
        min_mz, max_mz = np.min(all_mz), np.max(all_mz)
        mz_bins = np.arange(min_mz, max_mz + 0.1, 0.1)
        return mz_bins
    
    def spectrum_to_vector(self, spectrum: Dict, mz_bins: np.ndarray) -> np.ndarray:
        """将谱图转换为向量表示"""
        vector = np.zeros(len(mz_bins))
        for mz, intensity in zip(spectrum["m/z array"], spectrum["intensity array"]):
            idx = np.digitize(mz, mz_bins) - 1
            if 0 <= idx < len(mz_bins):
                vector[idx] += intensity
        return vector
    
    def compute_batch_similarity(self, batch_vectors: np.ndarray, 
                                target_vectors: np.ndarray) -> np.ndarray:
        """
        计算批次向量与目标向量之间的余弦相似度
        
        参数:
            batch_vectors: 批次向量矩阵 (n_batch, n_features)
            target_vectors: 目标向量矩阵 (n_target, n_features)
        
        返回:
            相似度矩阵 (n_batch, n_target)
        """
        # 归一化向量
        batch_norm = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
        target_norm = np.linalg.norm(target_vectors, axis=1, keepdims=True)
        
        # 避免除零
        batch_norm[batch_norm == 0] = 1
        target_norm[target_norm == 0] = 1
        
        batch_normalized = batch_vectors / batch_norm
        target_normalized = target_vectors / target_norm
        
        # 计算余弦相似度
        similarity = np.dot(batch_normalized, target_normalized.T)
        
        return similarity


# ============================================================================
# 3. 流式Transformer DNN相似度计算器
# ============================================================================

class StreamingTransformerDNNSimilarityCalculator:
    """流式Transformer DNN相似度计算器（从CSV读取嵌入向量，使用Tanimoto相似度）"""
    
    def __init__(self, embedding_file_path: str, max_samples: Optional[int] = None):
        """
        初始化流式Transformer DNN相似度计算器
        
        参数:
            embedding_file_path: 嵌入向量CSV文件路径
            max_samples: 最大处理样本数（用于测试，None表示处理全部数据）
        """
        self.embedding_file_path = embedding_file_path
        self.max_samples = max_samples
        
        # 获取总数据量
        self.total_count = self._count_rows()
        
        # 如果指定了max_samples，限制总数
        if self.max_samples is not None:
            self.total_count = min(self.total_count, self.max_samples)
        
        print(f"流式Transformer DNN相似度计算器初始化完成")
        print(f"  嵌入文件: {embedding_file_path}")
        print(f"  总数据量: {self.total_count}")
    
    def _count_rows(self) -> int:
        """统计CSV文件的行数"""
        with open(self.embedding_file_path, 'r', encoding='utf-8') as f:
            # Transformer.csv 没有表头，所有行都是数据
            return sum(1 for _ in f)
    
    def get_total_count(self) -> int:
        """获取总数据量"""
        return self.total_count
    
    def load_batch_embeddings(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        加载指定范围的嵌入向量批次
        
        参数:
            start_idx: 起始索引
            end_idx: 结束索引（不包含）
        
        返回:
            batch_embeddings: 嵌入向量矩阵 (n_samples, n_features)
        """
        # 限制end_idx不超过total_count
        end_idx = min(end_idx, self.total_count)
        
        # 读取指定行的数据
        # 注意：Transformer.csv 没有表头，第一列是嵌入向量的第一维（不是索引）
        # 索引从0开始：第0行是第一个数据，第1行是第二个数据，以此类推
        # 
        # 重要：start_idx 和 end_idx 是从原始数据文件来的，它们对应的是数据行的索引（从0开始）
        # 原始数据文件有表头，所以：
        #   - start_idx=0 对应原始文件的第2行（第1行是表头，第2行是第一个数据）
        #   - start_idx=1 对应原始文件的第3行（第1行是表头，第3行是第二个数据）
        # Transformer.csv 没有表头，所以：
        #   - start_idx=0 对应 Transformer.csv 的第0行（第一个数据）
        #   - start_idx=1 对应 Transformer.csv 的第1行（第二个数据）
        # 
        # skiprows=range(0, start_idx) 会跳过第0行到第start_idx-1行（共start_idx行）
        # 然后读取从第start_idx行开始的nrows行数据
        # 例如：start_idx=0, end_idx=10 -> 跳过0行，读取第0-9行（共10行）✓
        #      start_idx=10, end_idx=20 -> 跳过0-9行，读取第10-19行（共10行）✓
        # 不使用 index_col 和 header，因为第一列是数据，不是索引，也没有表头
        df = pd.read_csv(self.embedding_file_path, skiprows=range(0, start_idx), 
                        nrows=end_idx - start_idx, header=None)
        
        # 第一列是嵌入向量的第一维，不是索引，所以包含所有列
        # 所有列都是嵌入向量的维度
        # 由于没有表头，所有列都是数据列
        batch_embeddings = df.values.astype(np.float32)
        
        return batch_embeddings
    
    def compute_batch_similarity(self, batch_embeddings: np.ndarray,
                                 target_embeddings: np.ndarray) -> np.ndarray:
        """
        计算批次嵌入向量与目标嵌入向量之间的Tanimoto相似度
        
        参数:
            batch_embeddings: 批次嵌入向量矩阵 (n_batch, n_features)
            target_embeddings: 目标嵌入向量矩阵 (n_target, n_features)
        
        返回:
            相似度矩阵 (n_batch, n_target)
        
        注意:
            Tanimoto相似度公式（适用于二进制向量）: 
            T(A, B) = (A · B) / (||A||² + ||B||² - A · B)
            
            对于二进制向量（0/1），这个公式等价于Jaccard相似度。
            如果向量包含负值，此公式可能不适用，应该使用余弦相似度。
        """
        # 确保输入是float32类型
        batch_embeddings = batch_embeddings.astype(np.float32)
        target_embeddings = target_embeddings.astype(np.float32)
        
        # 计算点积矩阵 (n_batch, n_target)
        dot_product = np.dot(batch_embeddings, target_embeddings.T)
        
        # 计算每个向量的平方范数
        # batch_norm_sq: (n_batch, 1)
        batch_norm_sq = np.sum(batch_embeddings ** 2, axis=1, keepdims=True)
        # target_norm_sq: (1, n_target)
        target_norm_sq = np.sum(target_embeddings ** 2, axis=1, keepdims=True).T
        
        # 使用广播计算分母: ||A||² + ||B||² - A · B
        # batch_norm_sq 会广播到 (n_batch, n_target)
        # target_norm_sq 会广播到 (n_batch, n_target)
        denominator = batch_norm_sq + target_norm_sq - dot_product
        
        # 避免除零和数值不稳定
        # 如果分母非常小（接近0），说明两个向量都是零向量或几乎为零，相似度设为0
        # 使用一个小的epsilon来避免数值问题
        epsilon = 1e-10
        denominator = np.maximum(denominator, epsilon)
        
        # 计算Tanimoto相似度
        similarity = dot_product / denominator
        
        # 对于二进制向量，Tanimoto相似度应该在[0, 1]范围内
        # 但如果向量有负值，可能会超出这个范围，所以需要clip
        similarity = np.clip(similarity, 0.0, 1.0)
        
        # 处理特殊情况：如果两个向量都是零向量，相似度应该是0（而不是NaN）
        # batch_norm_sq是(n_batch, 1)，target_norm_sq是(1, n_target)
        # 需要创建(n_batch, n_target)的零向量掩码
        batch_zero = (batch_norm_sq < epsilon)  # (n_batch, 1)
        target_zero = (target_norm_sq < epsilon)  # (1, n_target)
        # 广播到(n_batch, n_target): batch_zero广播到列，target_zero广播到行
        # 使用np.logical_and确保正确的广播
        zero_mask = np.logical_and(batch_zero, target_zero)  # (n_batch, n_target)
        similarity[zero_mask] = 0.0
        
        return similarity


# ============================================================================
# 4. 流式融合网络构建器
# ============================================================================

class StreamingFusionNetworkBuilder:
    """流式融合网络构建器（增量构建，不存储完整相似度矩阵）"""
    
    def __init__(self, 
                 cosine_threshold: float = 0.7,
                 transformer_dnn_threshold: float = 0.7,
                 cosine_max_edges: int = 3,
                 transformer_dnn_max_edges: int = 3,
                 use_reciprocal: bool = True):
        """
        初始化流式融合网络构建器
        
        参数:
            cosine_threshold: Cosine相似度阈值
            transformer_dnn_threshold: Transformer DNN Tanimoto相似度阈值
            cosine_max_edges: 每个节点通过Cosine连接的最大边数
            transformer_dnn_max_edges: 每个节点通过Transformer DNN连接的最大边数
            use_reciprocal: 是否使用互惠连接（如果A->B满足条件，则同时添加B->A）
        """
        self.cosine_threshold = cosine_threshold
        self.transformer_dnn_threshold = transformer_dnn_threshold
        self.cosine_max_edges = cosine_max_edges
        self.transformer_dnn_max_edges = transformer_dnn_max_edges
        self.use_reciprocal = use_reciprocal
        
        # 创建空网络
        self.G = nx.Graph()
        
        print(f"流式融合网络构建器初始化完成")
        print(f"  Cosine阈值: {cosine_threshold}, 最大边数: {cosine_max_edges}")
        print(f"  Transformer DNN阈值: {transformer_dnn_threshold}, 最大边数: {transformer_dnn_max_edges}")
        print(f"  互惠连接: {use_reciprocal}")
    
    def add_nodes(self, batch_spectra: List[Dict], start_idx: int):
        """
        添加节点到网络
        
        参数:
            batch_spectra: 批次谱图数据列表
            start_idx: 起始索引
        """
        for i, spectrum in enumerate(batch_spectra):
            node_id = start_idx + i
            self.G.add_node(node_id, 
                          smiles=spectrum.get('smiles', ''),
                          precursor_mz=spectrum.get('PEPMASS', 0.0),
                          original_index=spectrum.get('original_index', node_id),
                          ion_mode=spectrum.get('ion_mode', 'unknown'),
                          precursor_type=spectrum.get('precursor_type', ''),
                          cluster_id=spectrum.get('cluster_id', ''))
    
    def add_edges_from_similarity(self, 
                                  batch_indices: np.ndarray,
                                  target_indices: np.ndarray,
                                  similarity_matrix: np.ndarray,
                                  edge_type: str,
                                  threshold: float,
                                  max_edges: int):
        """
        根据相似度矩阵添加边到网络
        
        参数:
            batch_indices: 批次节点索引数组
            target_indices: 目标节点索引数组
            similarity_matrix: 相似度矩阵 (n_batch, n_target)
            edge_type: 边类型（'cosine'或'transformer_dnn'）
            threshold: 相似度阈值
            max_edges: 每个节点的最大边数
        """
        n_batch = len(batch_indices)
        n_target = len(target_indices)
        
        for i in range(n_batch):
            batch_idx = batch_indices[i]
            
            # 检查当前节点已经有多少条该类型的边
            current_edges = list(self.G.edges(batch_idx, data=True))
            current_edge_type_count = sum(1 for e in current_edges 
                                         if edge_type in e[2].get('edge_type', '').lower())
            
            # 如果已经达到max_edges，跳过（不再添加该类型的边）
            if current_edge_type_count >= max_edges:
                continue
            
            # 获取当前节点的所有相似度
            similarities = similarity_matrix[i, :]
            
            # 找到满足阈值条件的相似度
            valid_mask = similarities >= threshold
            
            if not np.any(valid_mask):
                continue
            
            # 获取满足条件的相似度值及其索引
            valid_similarities = similarities[valid_mask]
            valid_target_indices = target_indices[valid_mask]
            
            # 按相似度降序排序
            sorted_indices = np.argsort(valid_similarities)[::-1]
            
            # 计算还可以添加多少条边
            remaining_slots = max_edges - current_edge_type_count
            top_k = min(remaining_slots, len(sorted_indices))
            top_indices = sorted_indices[:top_k]
            
            # 添加边
            for top_idx in top_indices:
                target_idx = valid_target_indices[top_idx]
                similarity = valid_similarities[top_idx]
                
                # 避免自连接
                if batch_idx == target_idx:
                    continue
                
                # 检查是否已经存在该类型的边（如果存在，只更新权重，不增加计数）
                if self.G.has_edge(batch_idx, target_idx):
                    current_edge_type = self.G[batch_idx][target_idx].get('edge_type', 'unknown').lower()
                    # 如果边已存在且类型相同，只更新权重
                    if edge_type in current_edge_type:
                        current_weight = self.G[batch_idx][target_idx].get('weight', 0.0)
                        self.G[batch_idx][target_idx]['weight'] = max(current_weight, similarity)
                    # 如果边已存在但类型不同
                    else:
                        # 策略：先用cosine连接，然后在此基础上连接transformer_dnn
                        # 如果已有cosine边，transformer_dnn只更新权重，保持cosine类型
                        if 'cosine' in current_edge_type and edge_type == 'transformer_dnn':
                            # 已有cosine边，transformer_dnn只更新权重，不改变类型
                            current_weight = self.G[batch_idx][target_idx].get('weight', 0.0)
                            self.G[batch_idx][target_idx]['weight'] = max(current_weight, similarity)
                            # 不增加计数，因为这是基于已有cosine边的增强
                        # 如果已有transformer_dnn边，添加cosine时改为cosine类型（cosine是基础）
                        elif 'transformer_dnn' in current_edge_type and edge_type == 'cosine':
                            # 将边类型改为cosine（因为cosine是基础连接）
                            self.G[batch_idx][target_idx]['edge_type'] = 'cosine'
                            current_weight = self.G[batch_idx][target_idx].get('weight', 0.0)
                            self.G[batch_idx][target_idx]['weight'] = max(current_weight, similarity)
                            # 不增加计数，因为这是替换类型
                        # 其他情况：如果还有空位，可以添加新类型（但这种情况应该很少）
                        elif current_edge_type_count < max_edges:
                            # 更新边类型（合并类型）- 保留此逻辑以防其他情况
                            self.G[batch_idx][target_idx]['edge_type'] = f"{current_edge_type}_{edge_type}"
                            current_weight = self.G[batch_idx][target_idx].get('weight', 0.0)
                            self.G[batch_idx][target_idx]['weight'] = max(current_weight, similarity)
                            current_edge_type_count += 1
                else:
                    # 添加新边
                    self.G.add_edge(batch_idx, target_idx,
                                  weight=similarity,
                                  edge_type=edge_type)
                    current_edge_type_count += 1
                
                # 如果使用互惠连接，添加反向边（但也要检查目标节点的限制）
                if self.use_reciprocal:
                    target_edges = list(self.G.edges(target_idx, data=True))
                    target_edge_type_count = sum(1 for e in target_edges 
                                                if edge_type in e[2].get('edge_type', '').lower())
                    
                    if not self.G.has_edge(target_idx, batch_idx):
                        if target_edge_type_count < max_edges:
                            self.G.add_edge(target_idx, batch_idx,
                                          weight=similarity,
                                          edge_type=edge_type)
                    else:
                        # 如果边已存在，只更新权重
                        current_edge_type = self.G[target_idx][batch_idx].get('edge_type', 'unknown').lower()
                        if edge_type not in current_edge_type and target_edge_type_count < max_edges:
                            self.G[target_idx][batch_idx]['edge_type'] = f"{current_edge_type}_{edge_type}"
                            current_weight = self.G[target_idx][batch_idx].get('weight', 0.0)
                            self.G[target_idx][batch_idx]['weight'] = max(current_weight, similarity)
    
    def enforce_edge_limits(self):
        """
        强制执行边数限制，确保每个节点每种方法最多max_edges条边
        保留相似度最高的边
        """
        print("\n[边数限制检查] 确保每个节点每种方法最多3条边...")
        
        from collections import defaultdict
        
        # 统计每个节点每种类型的边
        node_edge_counts = defaultdict(lambda: {'cosine': [], 'transformer_dnn': []})
        
        for u, v, data in self.G.edges(data=True):
            edge_type = str(data.get('edge_type', '')).lower()
            weight = data.get('weight', 0.0)
            
            # 判断边类型
            is_cosine = 'cosine' in edge_type and 'transformer_dnn' not in edge_type
            is_transformer_dnn = 'transformer_dnn' in edge_type and 'cosine' not in edge_type
            
            if is_cosine:
                node_edge_counts[u]['cosine'].append((v, weight))
                node_edge_counts[v]['cosine'].append((u, weight))
            elif is_transformer_dnn:
                node_edge_counts[u]['transformer_dnn'].append((v, weight))
                node_edge_counts[v]['transformer_dnn'].append((u, weight))
            else:
                # 混合类型，同时计入两种类型
                node_edge_counts[u]['cosine'].append((v, weight))
                node_edge_counts[v]['cosine'].append((u, weight))
                node_edge_counts[u]['transformer_dnn'].append((v, weight))
                node_edge_counts[v]['transformer_dnn'].append((u, weight))
        
        # 找出需要删除的边
        edges_to_remove = set()
        violations_fixed = 0
        
        for node_id, edge_lists in node_edge_counts.items():
            for edge_type, edges in edge_lists.items():
                max_edges = self.cosine_max_edges if edge_type == 'cosine' else self.transformer_dnn_max_edges
                
                if len(edges) > max_edges:
                    # 按权重降序排序
                    sorted_edges = sorted(edges, key=lambda x: x[1], reverse=True)
                    
                    # 保留前max_edges条，删除其余的
                    edges_to_delete = sorted_edges[max_edges:]
                    
                    for target, weight in edges_to_delete:
                        # 标记要删除的边（使用较小的节点ID作为第一个参数，确保一致性）
                        edge_key = (min(node_id, target), max(node_id, target))
                        edges_to_remove.add(edge_key)
                        violations_fixed += 1
        
        if len(edges_to_remove) > 0:
            print(f"  发现 {violations_fixed} 个违规，需要删除 {len(edges_to_remove)} 条边")
            
            # 删除超限的边
            for u, v in edges_to_remove:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
            
            print(f"  修复完成，当前边数: {self.G.number_of_edges()}")
        else:
            print("  所有节点都符合限制，无需修复")
    
    def export_network_to_csv(self, output_dir: str, prefix: str = "streaming_fusion_network"):
        """
        导出网络到CSV文件
        
        参数:
            output_dir: 输出目录
            prefix: 文件前缀
        
        返回:
            nodes_file: 节点文件路径
            edges_file: 边文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 导出节点
        nodes_data = []
        for node_id in self.G.nodes():
            node_data = self.G.nodes[node_id]
            nodes_data.append({
                'node_id': node_id,
                'smiles': node_data.get('smiles', ''),
                'precursor_mz': node_data.get('precursor_mz', 0.0),
                'original_index': node_data.get('original_index', node_id),
                'cluster_id': node_data.get('cluster_id', '')
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_file = os.path.join(output_dir, f"{prefix}_nodes.csv")
        nodes_df.to_csv(nodes_file, index=False, encoding='utf-8-sig')
        print(f"节点信息已保存到: {nodes_file} (共 {len(nodes_data)} 个节点)")
        
        # 导出边（包含离子模式信息）
        edges_data = []
        for u, v, data in self.G.edges(data=True):
            # 获取源节点和目标节点的离子模式和离子类型
            source_ion_mode = self.G.nodes[u].get('ion_mode', 'unknown')
            target_ion_mode = self.G.nodes[v].get('ion_mode', 'unknown')
            source_precursor_type = self.G.nodes[u].get('precursor_type', '')
            target_precursor_type = self.G.nodes[v].get('precursor_type', '')
            
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0.0),
                'edge_type': data.get('edge_type', 'unknown'),
                'source_ion_mode': source_ion_mode,
                'target_ion_mode': target_ion_mode,
                'source_precursor_type': source_precursor_type,
                'target_precursor_type': target_precursor_type
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_file = os.path.join(output_dir, f"{prefix}_edges.csv")
        edges_df.to_csv(edges_file, index=False, encoding='utf-8-sig')
        print(f"边信息已保存到: {edges_file} (共 {len(edges_data)} 条边)")
        
        return nodes_file, edges_file
    
    def build_streaming_fusion_network(self,
                                      csv_file: str,
                                      embedding_file: str,
                                      output_dir: str = "output",
                                      batch_size: int = 5000,
                                      max_samples: Optional[int] = None,
                                      cosine_threshold: float = 0.7,
                                      transformer_dnn_threshold: float = 0.7,
                                      cosine_max_edges: int = 3,
                                      transformer_dnn_max_edges: int = 3,
                                      use_reciprocal: bool = True,
                                      enable_optimization: bool = False,
                                      optimization_ratio: float = 0.1) -> nx.Graph:
        """
        构建流式融合网络
        
        参数:
            csv_file: CSV文件路径（包含谱图数据）
            embedding_file: 嵌入向量CSV文件路径
            output_dir: 输出目录
            batch_size: 批次大小
            max_samples: 最大处理样本数（用于测试）
            cosine_threshold: Cosine相似度阈值
            transformer_dnn_threshold: Transformer DNN Tanimoto相似度阈值
            cosine_max_edges: 每个节点通过Cosine连接的最大边数
            transformer_dnn_max_edges: 每个节点通过Transformer DNN连接的最大边数
            use_reciprocal: 是否使用互惠连接
        
        返回:
            G: 融合网络图
        """
        print("=" * 80)
        print("开始构建流式融合网络")
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化构建器
        self.__init__(cosine_threshold=cosine_threshold,
                     transformer_dnn_threshold=transformer_dnn_threshold,
                     cosine_max_edges=cosine_max_edges,
                     transformer_dnn_max_edges=transformer_dnn_max_edges,
                     use_reciprocal=use_reciprocal)
        
        # 初始化数据加载器和相似度计算器
        data_loader = StreamingDataLoader(csv_file, batch_size, max_samples)
        cosine_calculator = StreamingCosineSimilarityCalculator()
        transformer_dnn_calculator = StreamingTransformerDNNSimilarityCalculator(embedding_file, max_samples)
        
        total_count = data_loader.get_total_count()
        print(f"\n总数据量: {total_count}")
        print(f"批次大小: {batch_size}")
        print(f"预计批次数: {(total_count + batch_size - 1) // batch_size}\n")
        
        # 第一遍：计算m/z bins（需要先加载所有数据）
        print("[步骤 1/3] 计算m/z bins...")
        all_spectra = []
        valid_count = 0
        for start_idx in range(0, total_count, batch_size):
            end_idx = min(start_idx + batch_size, total_count)
            batch_spectra, _ = data_loader.load_batch(start_idx, end_idx)
            
            # 只保留有效的谱图（有m/z数据的）
            for spec in batch_spectra:
                mz_array = spec.get("m/z array", np.array([]))
                if len(mz_array) > 0:
                    all_spectra.append(spec)
                    valid_count += 1
            
            if len(all_spectra) >= 10000:  # 限制用于计算bins的样本数
                break
        
        print(f"  加载了 {valid_count} 个有效谱图用于计算m/z bins")
        
        if len(all_spectra) == 0:
            raise ValueError("错误: 没有找到有效的谱图数据（所有谱图的m/z数组都为空）")
        
        mz_bins = cosine_calculator.compute_mz_bins(all_spectra)
        cosine_calculator.mz_bins = mz_bins
        print(f"  m/z bins计算完成: {len(mz_bins)} 个bins (范围: {mz_bins[0]:.2f} - {mz_bins[-1]:.2f})")
        
        # 第二遍：流式构建网络
        print("\n[步骤 2/3] 流式构建网络...")
        
        num_batches = (total_count + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc="处理批次") as pbar:
            for batch_start in range(0, total_count, batch_size):
                batch_end = min(batch_start + batch_size, total_count)
                
                # 加载批次数据
                batch_spectra, _ = data_loader.load_batch(batch_start, batch_end)
                batch_indices = np.arange(batch_start, batch_end)
                
                # 添加节点
                self.add_nodes(batch_spectra, batch_start)
                
                # 转换为向量
                batch_vectors = np.array([cosine_calculator.spectrum_to_vector(s, mz_bins) 
                                        for s in batch_spectra])
                
                # 加载批次嵌入向量
                batch_embeddings = transformer_dnn_calculator.load_batch_embeddings(batch_start, batch_end)
                
                # 与之前的所有节点计算相似度
                if batch_start > 0:
                    # 加载之前的所有数据
                    prev_spectra = []
                    prev_indices = []
                    prev_vectors = []
                    prev_embeddings = []
                    
                    for prev_start in range(0, batch_start, batch_size):
                        prev_end = min(prev_start + batch_size, batch_start)
                        prev_batch_spectra, _ = data_loader.load_batch(prev_start, prev_end)
                        prev_batch_indices = np.arange(prev_start, prev_end)
                        prev_batch_vectors = np.array([cosine_calculator.spectrum_to_vector(s, mz_bins) 
                                                      for s in prev_batch_spectra])
                        prev_batch_embeddings = transformer_dnn_calculator.load_batch_embeddings(prev_start, prev_end)
                        
                        prev_spectra.extend(prev_batch_spectra)
                        prev_indices.extend(prev_batch_indices)
                        prev_vectors.append(prev_batch_vectors)
                        prev_embeddings.append(prev_batch_embeddings)
                    
                    if len(prev_vectors) > 0:
                        prev_vectors = np.vstack(prev_vectors)
                        prev_embeddings = np.vstack(prev_embeddings)
                        prev_indices = np.array(prev_indices)
                        
                        # 计算Cosine相似度
                        cosine_sim = cosine_calculator.compute_batch_similarity(batch_vectors, prev_vectors)
                        self.add_edges_from_similarity(batch_indices, prev_indices, cosine_sim,
                                                       'cosine', cosine_threshold, cosine_max_edges)
                        
                        # 计算Transformer DNN Tanimoto相似度
                        transformer_dnn_sim = transformer_dnn_calculator.compute_batch_similarity(
                            batch_embeddings, prev_embeddings)
                        self.add_edges_from_similarity(batch_indices, prev_indices, transformer_dnn_sim,
                                                       'transformer_dnn', transformer_dnn_threshold, 
                                                       transformer_dnn_max_edges)
                
                # 批次内部计算相似度
                if len(batch_spectra) > 1:
                    cosine_sim = cosine_calculator.compute_batch_similarity(batch_vectors, batch_vectors)
                    # 只使用上三角矩阵（避免重复）
                    cosine_sim = np.triu(cosine_sim, k=1)
                    self.add_edges_from_similarity(batch_indices, batch_indices, cosine_sim,
                                                   'cosine', cosine_threshold, cosine_max_edges)
                    
                    transformer_dnn_sim = transformer_dnn_calculator.compute_batch_similarity(
                        batch_embeddings, batch_embeddings)
                    transformer_dnn_sim = np.triu(transformer_dnn_sim, k=1)
                    self.add_edges_from_similarity(batch_indices, batch_indices, transformer_dnn_sim,
                                                   'transformer_dnn', transformer_dnn_threshold,
                                                   transformer_dnn_max_edges)
                
                pbar.update(1)
        
        print(f"\n[步骤 3/3] 保存网络...")
        
        # 先执行边数限制检查（确保每个节点每种方法最多max_edges条边）
        self.enforce_edge_limits()
        
        # 保存网络
        network_file = os.path.join(output_dir, "streaming_fusion_network.pkl")
        with open(network_file, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"  网络已保存到: {network_file}")
        
        # 导出为CSV
        nodes_file, edges_file = self.export_network_to_csv(output_dir, prefix="streaming_fusion_network")
        
        print(f"\n最终网络统计:")
        print(f"  节点数: {self.G.number_of_nodes()}")
        print(f"  边数: {self.G.number_of_edges()}")
        
        # 可选的后处理优化
        if enable_optimization:
            self.optimize_network_edges(
                csv_file=csv_file,
                embedding_file=embedding_file,
                cosine_calculator=cosine_calculator,
                transformer_dnn_calculator=transformer_dnn_calculator,
                mz_bins=mz_bins,
                data_loader=data_loader,
                batch_size=batch_size,
                max_samples=max_samples,
                cosine_threshold=cosine_threshold,
                transformer_dnn_threshold=transformer_dnn_threshold,
                cosine_max_edges=cosine_max_edges,
                transformer_dnn_max_edges=transformer_dnn_max_edges,
                optimization_ratio=optimization_ratio
            )
            # 重新保存优化后的网络
            network_file = os.path.join(output_dir, "streaming_fusion_network.pkl")
            with open(network_file, 'wb') as f:
                pickle.dump(self.G, f)
            nodes_file, edges_file = self.export_network_to_csv(output_dir, prefix="streaming_fusion_network")
        
        print("\n" + "=" * 80)
        print("完成！")
        print("=" * 80)
        
        return self.G
    
    def optimize_network_edges(self,
                               csv_file: str,
                               embedding_file: str,
                               cosine_calculator: StreamingCosineSimilarityCalculator,
                               transformer_dnn_calculator: StreamingTransformerDNNSimilarityCalculator,
                               mz_bins: np.ndarray,
                               data_loader: StreamingDataLoader,
                               batch_size: int = 5000,
                               max_samples: Optional[int] = None,
                               cosine_threshold: float = 0.7,
                               transformer_dnn_threshold: float = 0.7,
                               cosine_max_edges: int = 3,
                               transformer_dnn_max_edges: int = 3,
                               optimization_ratio: float = 0.1):
        """
        后处理优化：检查并补充遗漏的高相似度连接
        
        参数:
            optimization_ratio: 优化比例，只检查前optimization_ratio比例的节点（0.1表示检查前10%的节点）
            设置为1.0表示检查所有节点（可能很慢）
        
        注意：这是一个可选的后处理步骤，用于减少流式方法可能遗漏的连接
        """
        print("\n" + "=" * 80)
        print("开始后处理优化（检查遗漏的高相似度连接）")
        print("=" * 80)
        
        total_count = data_loader.get_total_count()
        if max_samples is not None:
            total_count = min(total_count, max_samples)
        
        # 只优化部分节点（避免计算量过大）
        optimize_count = int(total_count * optimization_ratio)
        print(f"优化节点数: {optimize_count} / {total_count} ({optimization_ratio*100:.1f}%)")
        
        optimized_edges = 0
        
        # 对每个节点，检查是否遗漏了高相似度连接
        for node_idx in tqdm(range(optimize_count), desc="优化节点"):
            # 获取当前节点的所有连接
            current_edges = list(self.G.edges(node_idx, data=True))
            current_edge_weights = {edge[1]: edge[2].get('weight', 0.0) for edge in current_edges}
            
            # 统计当前连接数（按类型）
            cosine_edges = [e for e in current_edges if 'cosine' in e[2].get('edge_type', '')]
            transformer_dnn_edges = [e for e in current_edges if 'transformer_dnn' in e[2].get('edge_type', '')]
            
            # 如果已经达到max_edges，检查是否有更高相似度的连接被遗漏
            if len(cosine_edges) >= cosine_max_edges or len(transformer_dnn_edges) >= transformer_dnn_max_edges:
                # 加载当前节点的数据
                batch_spectra, _ = data_loader.load_batch(node_idx, node_idx + 1)
                if len(batch_spectra) == 0:
                    continue
                
                node_spectrum = batch_spectra[0]
                node_vector = cosine_calculator.spectrum_to_vector(node_spectrum, mz_bins)
                node_embedding = transformer_dnn_calculator.load_batch_embeddings(node_idx, node_idx + 1)[0]
                
                # 检查与其他所有节点的相似度（采样检查，避免计算量过大）
                sample_size = min(1000, total_count)  # 每次只检查1000个其他节点
                sample_indices = np.random.choice(total_count, sample_size, replace=False)
                sample_indices = sample_indices[sample_indices != node_idx]  # 排除自己
                
                if len(sample_indices) == 0:
                    continue
                
                # 加载样本节点的数据
                sample_vectors = []
                sample_embeddings = []
                for sample_idx in sample_indices:
                    sample_spectra, _ = data_loader.load_batch(sample_idx, sample_idx + 1)
                    if len(sample_spectra) > 0:
                        sample_vectors.append(cosine_calculator.spectrum_to_vector(sample_spectra[0], mz_bins))
                        sample_embeddings.append(transformer_dnn_calculator.load_batch_embeddings(sample_idx, sample_idx + 1)[0])
                
                if len(sample_vectors) == 0:
                    continue
                
                sample_vectors = np.array(sample_vectors)
                sample_embeddings = np.array(sample_embeddings)
                
                # 计算相似度
                cosine_sim = cosine_calculator.compute_batch_similarity(
                    node_vector.reshape(1, -1), sample_vectors)[0]
                transformer_dnn_sim = transformer_dnn_calculator.compute_batch_similarity(
                    node_embedding.reshape(1, -1), sample_embeddings)[0]
                
                # 检查是否有更高相似度的连接
                for i, sample_idx in enumerate(sample_indices[:len(cosine_sim)]):
                    # Cosine相似度检查
                    if cosine_sim[i] >= cosine_threshold:
                        if sample_idx not in current_edge_weights or cosine_sim[i] > current_edge_weights[sample_idx]:
                            # 如果当前连接数未满，或者新连接相似度更高，则添加/替换
                            if len(cosine_edges) < cosine_max_edges:
                                self.G.add_edge(node_idx, sample_idx, weight=cosine_sim[i], edge_type='cosine')
                                optimized_edges += 1
                            elif cosine_sim[i] > min([e[2].get('weight', 0.0) for e in cosine_edges]):
                                # 替换最低相似度的连接
                                min_edge = min(cosine_edges, key=lambda e: e[2].get('weight', 0.0))
                                self.G.remove_edge(min_edge[0], min_edge[1])
                                self.G.add_edge(node_idx, sample_idx, weight=cosine_sim[i], edge_type='cosine')
                                optimized_edges += 1
                    
                    # Transformer DNN Tanimoto相似度检查
                    if transformer_dnn_sim[i] >= transformer_dnn_threshold:
                        if sample_idx not in current_edge_weights or transformer_dnn_sim[i] > current_edge_weights[sample_idx]:
                            if len(transformer_dnn_edges) < transformer_dnn_max_edges:
                                self.G.add_edge(node_idx, sample_idx, weight=transformer_dnn_sim[i], edge_type='transformer_dnn')
                                optimized_edges += 1
                            elif transformer_dnn_sim[i] > min([e[2].get('weight', 0.0) for e in transformer_dnn_edges]):
                                min_edge = min(transformer_dnn_edges, key=lambda e: e[2].get('weight', 0.0))
                                self.G.remove_edge(min_edge[0], min_edge[1])
                                self.G.add_edge(node_idx, sample_idx, weight=transformer_dnn_sim[i], edge_type='transformer_dnn')
                                optimized_edges += 1
        
        print(f"\n优化完成: 新增/替换了 {optimized_edges} 条边")
        print(f"最终网络统计:")
        print(f"  节点数: {self.G.number_of_nodes()}")
        print(f"  边数: {self.G.number_of_edges()}")
        print("=" * 80)


# ============================================================================
# 5. 主函数
# ============================================================================

def main_streaming(csv_file: str = "01_testdata/gnps_clean_testdata_originindex.csv",
                   embedding_file: str = "02_embedding/Transformer.csv",
                   output_dir: str = "./03_data_bulding/output",
                   batch_size: int = 5000,
                   max_samples: Optional[int] = None,
                   cosine_threshold: float = 0.7,
                   transformer_dnn_threshold: float = 0.90,
                   cosine_max_edges: int = 3,
                   transformer_dnn_max_edges: int = 3,
                   use_reciprocal: bool = True,
                   enable_optimization: bool = False,
                   optimization_ratio: float = 0.1):
    """
    流式融合分子网络构建主函数
    
    参数:
        csv_file: CSV文件路径（包含smiles列）
        embedding_file: 嵌入向量CSV文件路径
        output_dir: 输出目录
        batch_size: 批次大小
        max_samples: 最大处理样本数（用于测试，None表示处理全部数据）
        cosine_threshold: Cosine相似度阈值
        transformer_dnn_threshold: Transformer DNN Tanimoto相似度阈值
        cosine_max_edges: 每个节点通过Cosine连接的最大边数
        transformer_dnn_max_edges: 每个节点通过Transformer DNN连接的最大边数
        use_reciprocal: 是否使用互惠连接
        enable_optimization: 是否启用后处理优化（检查遗漏的高相似度连接）
        optimization_ratio: 优化比例，只检查前optimization_ratio比例的节点（0.1表示检查前10%）
    """
    builder = StreamingFusionNetworkBuilder()
    builder.build_streaming_fusion_network(
        csv_file=csv_file,
        embedding_file=embedding_file,
        output_dir=output_dir,
        batch_size=batch_size,
        max_samples=max_samples,
        cosine_threshold=cosine_threshold,
        transformer_dnn_threshold=transformer_dnn_threshold,
        cosine_max_edges=cosine_max_edges,
        transformer_dnn_max_edges=transformer_dnn_max_edges,
        use_reciprocal=use_reciprocal,
        enable_optimization=enable_optimization,
        optimization_ratio=optimization_ratio
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FMN benchmark network.")
    parser.add_argument("--spectra", required=True, help="Path to gnps_clean_testdata_originindex.csv")
    parser.add_argument("--fingerprints", required=True, help="Path to the headerless Transformer.csv")
    parser.add_argument("--output-dir", default="output", help="Directory for network outputs")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cosine-threshold", type=float, default=0.7)
    parser.add_argument("--transformer-threshold", type=float, default=0.9)
    parser.add_argument("--cosine-max-edges", type=int, default=3)
    parser.add_argument("--transformer-max-edges", type=int, default=3)
    parser.add_argument("--no-reciprocal", action="store_true")
    parser.add_argument("--enable-optimization", action="store_true")
    parser.add_argument("--optimization-ratio", type=float, default=0.1)
    args = parser.parse_args()

    main_streaming(
        csv_file=args.spectra,
        embedding_file=args.fingerprints,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        cosine_threshold=args.cosine_threshold,
        transformer_dnn_threshold=args.transformer_threshold,
        cosine_max_edges=args.cosine_max_edges,
        transformer_dnn_max_edges=args.transformer_max_edges,
        use_reciprocal=not args.no_reciprocal,
        enable_optimization=args.enable_optimization,
        optimization_ratio=args.optimization_ratio,
    )
