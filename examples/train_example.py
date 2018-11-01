import torch
from psychdr.data_loader import build_hetero_data
from psychdr.models import FeatureReducer, HGTWithDiseaseSim, LinkPredictor
# ... (省略部分 import)

def main():
    # 1. 用户只需修改这里的路径
    data_dir = "../data" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 调用包内函数建图
    data, id_map, nodes, edges = build_hetero_data(data_dir, ...)
    
    # 3. 初始化模型
    node_in_dims = {ntype: data[ntype].x.size(1) for ntype in data.node_types}
    model = HGTWithDiseaseSim(hidden_channels=256, metadata=data.metadata(), node_in_dims=node_in_dims).to(device)
    predictor = LinkPredictor(256).to(device)
    feature_reducer = FeatureReducer(...).to(device)
    
    # 4. 训练
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.001)
    for epoch in range(100):
        # loss = train_one_epoch(...)
        pass
        
    # 5. 保存模型权重
    torch.save({
        'model': model.state_dict(),
        'predictor': predictor.state_dict(),
        'reducer': feature_reducer.state_dict()
    }, "saved_models/psychdr_weights.pth")
    print("模型已保存！")

if __name__ == "__main__":
    main()