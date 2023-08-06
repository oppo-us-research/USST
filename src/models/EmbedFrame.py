"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url
import timm
import math
from functools import reduce
from operator import mul
import numpy as np



class FullyConvResNet(models.ResNet):
    """ Remove the pooling layers of ResNet to make it fully convolutional
    """
    def __init__(self, base='resnet18', img_size=64, pretrained=False, **kwargs):
        # Start with standard resnet18 defined here 
        super().__init__(block = models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=1000, **kwargs)

        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls[base])
            self.load_state_dict(state_dict)

        # compute the feature dimension
        dim = float(img_size)
        for n in range(5):
            dim = np.ceil(dim / 2.0)
        self.num_features = int((16 * 32) * dim**2)


    # Reimplementing forward pass. 
    def forward_features(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)  # (64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (64, H/4, W/4)

        x = self.layer1(x)
        x = self.layer2(x)  # (128, H/8, W/8)
        x = self.layer3(x)  # (256, H/16, W/16)
        x = self.layer4(x)  # (512, H/32, W/32)
        
        x = torch.flatten(x, start_dim=1)  # (self.num_features,)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., out_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.out_layer = out_layer() if out_layer is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x



class FrameEmbedding(nn.Module):
    """ Embed each frame into a low-dimensional feature"""
    def __init__(self, cfg, input_size=224):
        super(FrameEmbedding, self).__init__()
        self.input_size = input_size
        self.embed_dim = cfg.feat_dim
        self.freeze_backbone = getattr(cfg, 'freeze_backbone', False)
        self.backbone_name = cfg.backbone
        # visual prompt tuning
        self.vpt = None
        if hasattr(cfg, 'vpt'):
            self.freeze_backbone = True
            self.vpt = getattr(cfg, 'vpt')
        
        if self.backbone_name == 'vit/b16-224':
            self.backbone = timm.create_model('vit_base_patch16_224', img_size=self.input_size, pretrained=True, num_classes=0)
        elif self.backbone_name == 'resnet18':
            img_size = self.input_size + 2 * self.vpt['num_tokens'] if self.vpt else self.input_size
            pretrained = True if self.vpt else False  # if not vpt, we train resnet from scratch
            self.backbone = FullyConvResNet('resnet18', img_size=img_size, pretrained=pretrained)  # train from scratch
        else:
            raise NotImplementedError
        
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False  # fix the parameters
        
        if self.vpt and self.backbone_name == 'vit/b16-224':
            self.prompt_proj = nn.Identity()  # do not project the prompt
            self.prompt_dropout = nn.Dropout(self.vpt['dropout'])
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.vpt['prompt_len'], self.vpt['prompt_dim']))
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.backbone.patch_embed.patch_size, 1) + self.vpt['prompt_dim']))
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        if self.vpt and self.backbone_name == 'resnet18':
            self.prompt_norm = nn.Identity()
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.vpt['num_tokens'],
                    input_size + 2 * self.vpt['num_tokens']
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, input_size, 2 * self.vpt['num_tokens']
            ))
            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)
        
        # two-layers MLP
        self.mlp = Mlp(self.backbone.num_features, hidden_features=512, out_features=self.embed_dim)
    
    
    def _pos_embed(self, x):
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.pos_embed
        return self.backbone.pos_drop(x)
    
    
    def incorporate_vit_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.backbone.patch_embed(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = self._pos_embed(x)
        # insert the learnable prompts
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x
    
    
    def incorporate_conv_prompt(self, x):
        B = x.shape[0]
        prompt_emb_lr = self.prompt_norm(
            self.prompt_embeddings_lr).expand(B, -1, -1, -1)
        prompt_emb_tb = self.prompt_norm(
            self.prompt_embeddings_tb).expand(B, -1, -1, -1)

        x = torch.cat((
            prompt_emb_lr[:, :, :, :self.vpt['num_tokens']],
            x, prompt_emb_lr[:, :, :, self.vpt['num_tokens']:]
            ), dim=-1)
        x = torch.cat((
            prompt_emb_tb[:, :, :self.vpt['num_tokens'], :],
            x, prompt_emb_tb[:, :, self.vpt['num_tokens']:, :]
        ), dim=-2)  # (B, 3, crop_size + num_prompts, crop_size + num_prompts)
        return x


    def forward(self, x):
        if self.vpt and self.backbone_name == 'vit/b16-224':
            # prompt tuning
            x = self.incorporate_vit_prompt(x)
            x = self.backbone.blocks(x)
            x = self.backbone.norm(x)
            x = x[:, 0]
        elif self.vpt and self.backbone_name == 'resnet18':
            # prompt tuning
            x = self.incorporate_conv_prompt(x)
            x = self.backbone.forward_features(x)  # (B, D)
        else:
            # cnn backbone
            x = self.backbone.forward_features(x)  # (B, D)
        x = self.mlp(x)
        return x