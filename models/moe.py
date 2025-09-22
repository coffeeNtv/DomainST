import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import csv

class ExpertMLB(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim):
        super(ExpertMLB, self).__init__()
        self.image_fc = nn.Linear(d_img, hidden_dim)
        self.text_fc = nn.Linear(d_text, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, image_feat, text_feat):
        image_proj = torch.tanh(self.image_fc(image_feat))                         # [n, hidden_dim]
        text_proj = torch.tanh(self.text_fc(text_feat.mean(0, keepdim=True)))     # [1, hidden_dim]
        text_proj = text_proj.expand(image_proj.size(0), -1)                       # [n, hidden_dim]
        fused = image_proj * text_proj                                             # Hadamard product
        output = self.fc(fused)
        return output


class ExpertMFB1(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim, k=5):
        super(ExpertMFB, self).__init__()
        self.k = k  # factor number
        self.o = hidden_dim  # output dimension

        self.image_fc = nn.Linear(d_img, self.o * self.k)
        self.text_fc = nn.Linear(d_text, self.o * self.k)
        self.dropout = nn.Dropout(0.1)

        self.output_fc = nn.Sequential(
            nn.Linear(self.o, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, image_feat, text_feat):
        img_proj = self.image_fc(image_feat)                                      # [n, o*k]
        txt_proj = self.text_fc(text_feat.mean(0, keepdim=True))                 # [1, o*k]
        txt_proj = txt_proj.expand(img_proj.size(0), -1)                         # [n, o*k]
        joint_repr = img_proj * txt_proj                                         # [n, o*k]
        joint_repr = self.dropout(joint_repr)

        # Reshape and sum-pool
        joint_repr = joint_repr.view(-1, self.o, self.k).sum(2)                  # [n, o]

        # Power + l2 norm
        joint_repr = torch.sign(joint_repr) * torch.sqrt(torch.abs(joint_repr) + 1e-10)
        joint_repr = F.normalize(joint_repr, dim=-1)

        output = self.output_fc(joint_repr)                                      # [n, hidden_dim]
        return output

class ExpertMFB(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim, k=5, r=5):
        super(ExpertMFB, self).__init__()
        self.k, self.r = k, r
        self.hidden_dim = hidden_dim
        self.image_fc = nn.Identity()
        self.text_fc = nn.Identity()
        if d_img != hidden_dim:
            self.image_fc = nn.Linear(d_img, hidden_dim)
        if d_text != hidden_dim:
            self.text_fc = nn.Linear(d_text, hidden_dim)

        self.img_proj = nn.Linear(hidden_dim, k * r)
        self.text_proj = nn.Linear(hidden_dim, k * r)

        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, image_feat, text_feat):
        image_feat = self.image_fc(image_feat)                       # [n, hidden_dim]
        text_feat = self.text_fc(text_feat.mean(dim=0, keepdim=True))  # [1, hidden_dim]
        text_feat = text_feat.expand(image_feat.size(0), -1)            # [n, hidden_dim]

        img_proj = self.img_proj(image_feat)                         # [n, k*r]
        text_proj = self.text_proj(text_feat)                        # [n, k*r]
        joint = img_proj * text_proj                                 # [n, k*r]
        joint = joint.view(-1, self.k, self.r)                       # [n, k, r]
        joint = joint.sum(dim=2)                                     # [n, k]
        joint = torch.sqrt(F.relu(joint)) - torch.sqrt(F.relu(-joint))  # signed sqrt
        joint = F.normalize(joint, p=2, dim=1)                       # l2 norm
        joint = self.dropout(joint)
        output = self.fc(joint)                                      # [n, hidden_dim]
        return output


class ExpertHadamard(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim):
        super(ExpertHadamard, self).__init__()
        self.image_fc = nn.Identity()
        self.text_fc = nn.Identity()
        if d_img != hidden_dim:
            self.image_fc = nn.Linear(d_img, hidden_dim)
        if d_text != hidden_dim:
            self.text_fc = nn.Linear(d_text, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, image_feat, text_feat):
        image_feat = self.image_fc(image_feat)                       # [n, hidden_dim]
        text_feat = self.text_fc(text_feat.mean(dim=0, keepdim=True))  # [1, hidden_dim]
        text_feat = text_feat.expand(image_feat.size(0), -1)            # [n, hidden_dim]
        combined_feat = image_feat * text_feat                       # [n, hidden_dim]
        output = self.fc(combined_feat)                              # [n, hidden_dim]
        return output


class ExpertConcatenate(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim):
        super(ExpertConcatenate, self).__init__()
        self.image_fc = nn.Identity()
        self.text_fc = nn.Identity()
        if d_img != hidden_dim:
            self.image_fc = nn.Linear(d_img, hidden_dim)
        if d_text != hidden_dim:
            self.text_fc = nn.Linear(d_text, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, image_feat, text_feat):
        # align dim
        image_feat = self.image_fc(image_feat)  # [n, hidden_dim]
        # average gene text feat to batch size
        text_feat = self.text_fc(text_feat.mean(dim=0, keepdim=True))  # [1, hidden_dim]
        text_feat = text_feat.expand(image_feat.size(0), -1)           # [n, hidden_dim]
        combined_feat = torch.cat([image_feat, text_feat], dim=1)      # [n, hidden_dim * 2]
        output = self.fc(combined_feat)                                # [n, hidden_dim]
        return output

class ExpertLinear(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim):
        super(ExpertLinear, self).__init__()
        # align dim
        self.image_fc = nn.Identity()
        self.text_fc = nn.Identity()
        if d_img != hidden_dim:
            self.image_fc = nn.Linear(d_img, hidden_dim)
        if d_text != hidden_dim:
            self.text_fc = nn.Linear(d_text, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, image_feat, text_feat):
        image_feat = self.image_fc(image_feat)                         # [n, hidden_dim]
        text_feat = self.text_fc(text_feat.mean(dim=0, keepdim=True))  # [1, hidden_dim]
        text_feat = text_feat.expand(image_feat.size(0), -1)           # [n, hidden_dim]
        combined_feat = image_feat + text_feat                         # [n, hidden_dim]
        output = self.fc(combined_feat)                                # [n, hidden_dim]
        return output

class ExpertAttention(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim, num_heads=4):
        super(ExpertAttention, self).__init__()
        # align dim
        self.image_fc = nn.Identity()
        self.text_fc = nn.Identity()
        if d_img != hidden_dim:
            self.image_fc = nn.Linear(d_img, hidden_dim)
        if d_text != hidden_dim:
            self.text_fc = nn.Linear(d_text, hidden_dim)
        # MHA
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, image_feat, text_feat):
        image_feat = self.image_fc(image_feat).unsqueeze(1)            # [n, 1, hidden_dim]
        text_feat = self.text_fc(text_feat).unsqueeze(0).expand(image_feat.size(0), -1, -1)  # [n, m, hidden_dim]
        attn_output, _ = self.attention(query=image_feat, key=text_feat, value=text_feat)    # [n, 1, hidden_dim]
        attn_output = attn_output.squeeze(1)                                                # [n, hidden_dim]
        output = self.fc(attn_output)                                                       # [n, hidden_dim]
        return output


class GatingNetwork(nn.Module):
    def __init__(self, d_img, hidden_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_img, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=1)
        )
    
    def forward(self, image_feat):
        gating_weights = self.fc(image_feat)  # [n, num_experts]
        return gating_weights


class MoEModel(nn.Module):
    def __init__(self, d_img, d_text, hidden_dim, num_experts, output_dim):
        super(MoEModel, self).__init__()
        self.num_experts = num_experts
        # define expert
        self.experts = nn.ModuleList([
            ExpertConcatenate(d_img, d_text, hidden_dim),
            ExpertLinear(d_img, d_text, hidden_dim),
            ExpertAttention(d_img, d_text, hidden_dim)
            #ExpertMLB(d_img, d_text, hidden_dim),
            #ExpertMFB(d_img, d_text, hidden_dim),
            #ExpertMFB1(d_img, d_text, hidden_dim),

            #ExpertHadamard(d_img, d_text, hidden_dim)
            # add more experts
        ])
        # gating network
        self.gating_network = GatingNetwork(d_img, hidden_dim, num_experts)
        # output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # === 控制是否保存 gating weights ===
        self.save_weights = True  # 默认不保存
        self.csv_path = "./gating_weights.csv"  # 默认保存路径

        # 初始化：如果设置保存，且文件存在则删除旧文件
        if self.save_weights and os.path.exists(self.csv_path):
            os.remove(self.csv_path)
    
    def forward(self, image_feat, text_feat):
        gating_weights = self.gating_network(image_feat)              # [n, num_experts]
        # === 保存 gating weights 到 CSV（如果启用） ===
        if self.save_weights:
            weights_np = gating_weights.detach().cpu().numpy()  # [n, num_experts]

            # 如果文件不存在，先写入 header
            if not os.path.exists(self.csv_path):
                columns = [f'Expert_{i}' for i in range(self.num_experts)]
                df = pd.DataFrame(columns=columns)
                df.to_csv(self.csv_path, index=False)

            # 追加写入当前 batch 的 gating weights
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(weights_np)
        # === MoE 融合过程 ===
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(image_feat, text_feat)             # [n, hidden_dim]
            expert_outputs.append(expert_output.unsqueeze(2))         # [n, hidden_dim, 1]
        # stack the output of each expert
        expert_outputs = torch.cat(expert_outputs, dim=2)             # [n, hidden_dim, num_experts]
        expert_outputs = expert_outputs.permute(0, 2, 1)              # [n, num_experts, hidden_dim]
        # weighted sum of expert output
        gating_weights = gating_weights.unsqueeze(1)                  # [n, 1, num_experts]
        fused_output = torch.bmm(gating_weights, expert_outputs).squeeze(1)  # [n, hidden_dim]
        output = self.output_layer(fused_output)                      # [n, output_dim]
        return output