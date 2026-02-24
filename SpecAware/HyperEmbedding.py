# Adaptive HyperEmbedding for Multi-sensor HSI via HyperNet
# Some codes from DOFA and Copernicus-FM, many thanks!

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpectralEmbedding import SpectralEmbedding, SpectralAwareTransformer
from timm.layers import trunc_normal_


class MatrixGenerator(nn.Module):
    # Generate U, V matrices
    def __init__(self, wv_planes=128, embed_dim=1024, kernel_size=8, rank=64):
        super().__init__()
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.patch_dim = kernel_size * kernel_size
        self.rank = rank

        self.u_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes, wv_planes//2),
            nn.GELU(),
            nn.Linear(wv_planes//2, embed_dim * rank)  # [E*r]
        )

        self.v_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes, wv_planes // 2),
            nn.GELU(),
            nn.Linear(wv_planes // 2, rank * self.patch_dim)  # [r*k²]
        )

        # bias
        self.bias_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes // 2),
            nn.GELU(),
            nn.Linear(wv_planes // 2, embed_dim)  # E
        )


    def forward(self, spectral_features):
        # input: [B, C, wv_planes]
        B, C, _ = spectral_features.shape

        u_params = self.u_generator(spectral_features)
        v_params = self.v_generator(spectral_features)

        U_matrices = u_params.view(B, C, self.embed_dim, self.rank)
        V_matrices = v_params.view(B, C, self.rank, self.patch_dim)

        sample_features = spectral_features.mean(dim=1)  # [B, wv_planes]
        bias = self.bias_generator(sample_features)  # [B, embed_dim]

        return U_matrices, V_matrices, bias


class ContentFeatureExtractor(nn.Module):
    # Extract content features from HSI patch simply
    def __init__(self, img_size=224, kernel_size=16, wv_planes=128):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes

        # patch pooling
        self.patch_pool_avg = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)
        self.patch_pool_max = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        expected_patch_num = (img_size // kernel_size) ** 2  # 224//8=28

        self.avg_feature_encoder = nn.Sequential(
            nn.Linear(expected_patch_num, expected_patch_num // 4),
            nn.GELU(),
            nn.Linear(expected_patch_num // 4, wv_planes)
        )

        self.max_feature_encoder = nn.Sequential(
            nn.Linear(expected_patch_num, expected_patch_num // 4),
            nn.GELU(),
            nn.Linear(expected_patch_num // 4, wv_planes)
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(wv_planes * 2, wv_planes * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes * 2, wv_planes),
        )

        self.layer_norm = nn.LayerNorm(wv_planes)

    def forward(self, x):
        B, C, H, W = x.shape

        x_pool_avg = self.patch_pool_avg(x) 
        x_pool_max = self.patch_pool_max(x) 

        x_avg_flat = x_pool_avg.view(B, C, -1)
        x_max_flat = x_pool_max.view(B, C, -1)

        avg_features = self.avg_feature_encoder(x_avg_flat)
        max_features = self.max_feature_encoder(x_max_flat)

        combined_features = torch.cat([avg_features, max_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.layer_norm(fused_features)

        return fused_features




class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # projection
        self.modal1_proj = nn.Linear(dim, dim)
        self.modal2_proj = nn.Linear(dim, dim)
        
        # fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.use_gating = False
        if self.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(2*dim, dim),
                nn.Sigmoid()
            )
    
    def forward(self, feat1, feat2):
        """
        feat1, feat2: [B, C, dim]
        """
        f1 = self.modal1_proj(feat1)
        f2 = self.modal2_proj(feat2)
        
        combined = torch.cat([f1, f2], dim=-1)
        
        fused = self.fusion(combined)
        
        if self.use_gating:
            gate_weight = self.gate(combined)
            output = gate_weight * fused + (1 - gate_weight) * f1
        else:
            output = fused + f1
        
        return output


class HyperEmbedding(nn.Module):
    def __init__(self, img_size=224, wv_planes=128, kernel_size=16, num_heads=4,
                 num_layers=1, embed_dim=768, rank=64):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.rank = rank

        # spectral embedding
        self.spectral_embedding = SpectralEmbedding(wv_planes)

        self.spectral_aware_transformer = SpectralAwareTransformer(
            input_dim=wv_planes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )

        self.content_feature_extractor = ContentFeatureExtractor(
            img_size=img_size,
            kernel_size=kernel_size,
            wv_planes=wv_planes
        )

        self.matrix_generator = MatrixGenerator(
            wv_planes=wv_planes,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            rank=rank
        )

        self.feature_fusion = CrossModalFusion(wv_planes)
        self.scaler = nn.Parameter(torch.tensor(0.02))
        
        self._init_weights()


    def _init_weights(self):
        bias_gen_output_head = self.matrix_generator.bias_generator[-1]
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is bias_gen_output_head:
                    nn.init.constant_(m.weight, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def matrix_computation(self, patches, U_matrices, V_matrices):
        """
        patches: [B, N, C×k²]
        U_matrices: [B, C, embed_dim, rank]  
        V_matrices: [B, C, rank, k²]
        return: [B, N, embed_dim]
        """
        B, N, _ = patches.shape
        C = U_matrices.shape[1]
        patch_dim = self.kernel_size * self.kernel_size
        
        patches_reshaped = patches.view(B, N, C, patch_dim)
    
        intermediate = torch.einsum('bncp,bcrp->bncr', 
                                    patches_reshaped, 
                                    V_matrices)
        
        channel_outputs = torch.einsum('bncr,bcer->bnce', 
                                        intermediate, 
                                        U_matrices)
        
        output = channel_outputs.sum(dim=2)  # [B, N, embed_dim]
    
        return output

    def forward(self, x, wavelengths, fwhm, sensor_name, data_level):
        B, C, H, W = x.shape

        if isinstance(wavelengths, np.ndarray):
            wavelengths = torch.from_numpy(wavelengths).float()

        # Spectral Embedding
        meta_embedding = self.spectral_embedding(wavelengths, fwhm, sensor_name, data_level)
        enhanced_meta_embedding = self.spectral_aware_transformer(meta_embedding)

        # Content Feature Extraction
        content_features = self.content_feature_extractor(x)

        # Fusion
        enhanced_features = self.feature_fusion(
            enhanced_meta_embedding, content_features) + enhanced_meta_embedding

        # Unfold
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.kernel_size)
        patches = patches.permute(0, 2, 1)

        # Matrix Computation
        U_matrices, V_matrices, bias = self.matrix_generator(enhanced_features)
        output = self.matrix_computation(patches, U_matrices, V_matrices)  # [B, N, embed_dim]

        output = output + bias.unsqueeze(1)
        output = output * self.scaler
        
        return output, enhanced_meta_embedding

