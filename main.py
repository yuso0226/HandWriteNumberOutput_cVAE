import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----- モデル定義 -----
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_latent))
        output = torch.sigmoid(self.fc_out(hidden))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, label)
        return x_recon, mu, logvar

# ----- Streamlit アプリケーション -----
st.title("🧠 条件付きVAEによる手書き数字の生成")

# モデル・デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(latent_dim=3, num_classes=10).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# ユーザー入力
digit = st.slider("生成したい数字（0〜9）", 0, 9, 0)
num_samples = st.slider("生成する枚数", 1, 16, 6)
latent_dim = 3

# 生成処理
if st.button("生成する"):
    z = torch.randn(num_samples, latent_dim).to(device)
    labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
    with torch.no_grad():
        x_gen = model.decoder(z, labels)

    # 画像表示
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        img_np = x_gen[i].squeeze().cpu().numpy()
        axes[i].imshow(img_np, cmap='gray')
        axes[i].axis("off")
    st.pyplot(fig)
