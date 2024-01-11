from typing import Optional, Set, Union
import torch.nn as nn
import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet



class StaticEmbedding(nn.Module):
    def __init__(self, max_input_size, hidden_channels, padding_idx=0):
        super(StaticEmbedding, self).__init__()
        self.hidden_channels = hidden_channels
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(max_input_size, self.hidden_channels, padding_idx=self.padding_idx)

    def forward(self, z):
        embedded_z = self.embedding(z)
        return embedded_z


## 首先schnet模型处理的是蛋白质的三维的结构，继承的是Schnet
class SchNetModel(SchNet): ##SchNet的模型是在import里面用过来的
    def __init__(
        self,
        max_len,
        hidden_channels: int = 128, #定义隐藏层的通道数
        out_dim: int = 6, #指定模型的输出维度
        num_filters: int = 128, #卷积层使用的过滤器的数量
        num_layers: int = 6, #卷积层的数量
        num_gaussians: int = 50,#定义用于径向过滤器的高斯函数数量
        cutoff: float = 10,#设置相互作用的截止距离。这个值决定了模型考虑原子间相互作用的最大距离。
        max_num_neighbors: int = 32,#定义考虑的最大邻居原子，个值限制了每个原子周围考虑的邻居
        readout: str = "add", #"add"（加法池化）。
        dipole: bool = False, #指定是否计算偶极矩。
        mean: Optional[float] = None, #定义标准化过程中使用的平均值。
        std: Optional[float] = None, #定义标准化过程中使用的标准差。
        atomref: Optional[torch.Tensor] = None, #提供原子参考值。这通常用于计算化学势能或类似的属性。
       
    ):
        """
        Initializes an instance of the SchNetModel class with the provided
        parameters.

        :param hidden_channels: Number of channels in the hidden layers
            (default: ``128``)
        :type hidden_channels: int
        :param out_dim: Output dimension of the model (default: ``1``)
        :type out_dim: int
        :param num_filters: Number of filters used in convolutional layers
            (default: ``128``)
        :type num_filters: int
        :param num_layers: Number of convolutional layers in the model
            (default: ``6``)
        :type num_layers: int
        :param num_gaussians: Number of Gaussian functions used for radial
            filters (default: ``50``)
        :type num_gaussians: int
        :param cutoff: Cutoff distance for interactions (default: ``10``)
        :type cutoff: float
        :param max_num_neighbors: Maximum number of neighboring atoms to
            consider (default: ``32``)
        :type max_num_neighbors: int
        :param readout: Global pooling method to be used (default: ``"add"``)
        :type readout: str
        """
        super().__init__(
            hidden_channels,
            num_filters,
            num_layers,
            num_gaussians,
            cutoff,  # None, # Interaction graph is not used
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        self.max_len = max_len
        self.readout = readout
        # Overwrite embbeding
        self.embedding =StaticEmbedding(max_len,hidden_channels,padding_idx=0)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)  #定义线性全连接层然后输出


    @property
    def required_batch_attributes(self) -> Set[str]: 
        """
        Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`) n是节点的数量，d是节点的维度
        - ``pos``: Node positions (shape: :math:`(n, 3)`) n是节点的数量，3是空间的坐标
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`) 边缘索引，e是边的数量
        - ``batch``: Batch indices (shape: :math:`(n,)`) n是节点的数量

        :return: Set of required batch attributes
        :rtype: Set[str]
        这个定义就是一个检查点，进行检查输入的数据是否符合规则的
        """
        return {"pos", "edge_index", "x", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]):
        #batch的类型 可以是batch 也可以是proteinbatch的类型
        """Implements the forward pass of the SchNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        h = self.embedding(batch.x) #蛋白质的节点就是残基的特征，提高特征的维度，添加非线性
        #这个在之前是有定义的
        edge_attr = self.distance_expansion(batch.edge_weight) #生成边缘属性



        #self.interaction是继承父类的SchNet，交互层中有着其他的神经网路，给
        #每一个节点h添加额外的属性信息，包含边缘信息，边的权重，边的因素等信息
        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, batch.edge_weight, edge_attr)

        h = self.lin1(h) #线性层，可能在父类，或者是其他的定义了的类中
        h = self.act(h) # 激活函数层 一样的在父类，或者是其他的定义的类中
        h = self.lin2(h)  #这个在类中已经有了定义的
        



        out = {
                "node_embedding": h,#每个节点的嵌入，其形状为 (|V|, d)，|V| 是节点数，d 是嵌入的维度。
                "graph_embedding": torch_scatter.scatter(
                    h, batch.batch, dim=0, reduce=self.readout
                    #graph_embedding：整个图的嵌入，通过 torch_scatter.scatter 函数实现，
                    #它根据 batch.batch（每个节点所属图的索引）对节点嵌入进行聚合，
                    #形状为 (n, d)，其中 n 是图的数量。
                ),
            }
        

        return out
            
        
        
        
        
        


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from graphein.protein.tensor.data import get_random_protein

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "encoder" / "schnet.yaml"
    )
    print(cfg)
    encoder = hydra.utils.instantiate(cfg.schnet)
    print(encoder)
    batch = ProteinBatch().from_protein_list(
        [get_random_protein() for _ in range(4)], follow_batch=["coords"]
    )
    batch.batch = batch.coords_batch
    batch.edges("knn_8", cache="edge_index")
    batch.pos = batch.coords[:, 1, :]
    batch.x = batch.residue_type
    print(batch)
    out = encoder.forward(batch)
    print(out)
