import torch.nn as nn


class MlpHead(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bn_layer=None, act_layer=nn.GELU, drop=0., out_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = bn_layer(hidden_features) if bn_layer is not None else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = bn_layer(out_features) if bn_layer is not None else nn.Identity()
        self.drop = nn.Dropout(drop)
        self.out_layer = out_layer() if out_layer is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.drop(x)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


class UncertaintyHead(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, cfg=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        
        # parse the configurations
        self.act_out = getattr(cfg, 'act_out', None)
        self.unct_type = getattr(cfg, 'type', 'spacetime')  # spacetime, temporal, hybrid
        
        out_dims = {'spacetime': 3, 'hybrid': 2, 'temporal': 1}
        assert self.unct_type in out_dims, "Invalid uncertainty type."
        
        self.fc2 = nn.Linear(hidden_features, out_dims[self.unct_type], bias=False)
        # output activation
        self.out_layer = nn.Softplus(beta=1) if self.act_out else None
        # initialize
        self.apply(self.weights_init)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.0001)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.out_layer:
            x = self.out_layer(x)
        return x
