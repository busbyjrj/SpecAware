import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

class MatrixGenerator(nn.Module):
    # Generate U, V matrices
    def __init__(self, wv_planes=128, decoder_dim=512, kernel_size=8, rank=64):
        super().__init__()
        self.wv_planes = wv_planes
        self.decoder_dim = decoder_dim
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
            nn.Linear(wv_planes//2, self.patch_dim * rank)  # P² × R
        )

        self.v_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(wv_planes, wv_planes//2),
            nn.GELU(),
            nn.Linear(wv_planes//2, rank * decoder_dim)  # R × D
        )

        self.bias_generator = nn.Sequential(
            nn.LayerNorm(wv_planes),
            nn.Linear(wv_planes, wv_planes//2),
            nn.GELU(),
            nn.Linear(wv_planes//2, self.patch_dim)  # P²
        )


    def forward(self, spectral_features):
        # input: [B, C, wv_planes]
        B, C, _ = spectral_features.shape

        u_weights = self.u_generator(spectral_features)
        v_weights = self.v_generator(spectral_features)

        U_matrices = u_weights.view(B, C, self.patch_dim, self.rank)
        V_matrices = v_weights.view(B, C, self.rank, self.decoder_dim)

        bias = self.bias_generator(spectral_features)
        bias = bias.view(B, C, self.patch_dim)

        return U_matrices, V_matrices, bias


class TransformerContentExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def forward(self, query, key, value):
        B, L, D = key.shape

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        atten_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        atten_weights = F.softmax(atten_weights, dim=-1)

        output = torch.bmm(atten_weights, v)

        return output, atten_weights




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
        
        # optional simple gating
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


class HyperLinear(nn.Module):
    def __init__(self,
                 wv_planes=128,
                 decoder_dim=512,
                 kernel_size=8,
                 rank=64):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.patch_dim = kernel_size * kernel_size
        self.rank = rank

        self.x_attention_pool = TransformerContentExtractor(
            input_dim=decoder_dim,
            hidden_dim=64,
        )

        self.x_feature_extractor = nn.Sequential(
            nn.Linear(decoder_dim, wv_planes),
            nn.GELU(),
            nn.Linear(wv_planes, wv_planes)
        )

        self.matrix_generator = MatrixGenerator(
            wv_planes=wv_planes,
            decoder_dim=decoder_dim,
            kernel_size=kernel_size,
            rank=rank
        )

        self.feature_fusion = CrossModalFusion(wv_planes)

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


    def matrix_computation(self, x, U_matrices, V_matrices):
        x_expanded = x.unsqueeze(2)  # [B, L, 1, D]

        intermediate = torch.einsum('blcd,bcrd->blcr', x_expanded, V_matrices)  # [B, L, C, R]

        output = torch.einsum('blcr,bcpr->blcp', intermediate, U_matrices)  # [B, L, C, P²]

        return output

    def forward(self, x, wavelengths_embedding):
        B, L, D = x.shape
        B1, C, _ = wavelengths_embedding.shape

        assert B == B1, "Batch size of x and wavelengths_embedding must be the same"

        # extract content features
        query = x.mean(dim=1, keepdim=True)  # [B, 1, D] - Global context
        attended_x, attention_weights = self.x_attention_pool(query, x, x)
        x_features = self.x_feature_extractor(attended_x.squeeze(1))  # [B, wv_planes]

        # expand the content features
        x_features_expanded = x_features.unsqueeze(1).expand(-1, C, -1)  # [B, C, wv_planes]
        generator_input = self.feature_fusion(wavelengths_embedding, x_features_expanded) + x_features_expanded

        # generate the U, V matrices and bias
        U_matrices, V_matrices, bias = self.matrix_generator(generator_input)
        output = self.matrix_computation(x, U_matrices, V_matrices)

        # bias
        output = output + bias.unsqueeze(1)
        output = output.permute(0, 1, 3, 2).contiguous()
        final_output = output.reshape(B, L, self.patch_dim * C)

        return final_output
