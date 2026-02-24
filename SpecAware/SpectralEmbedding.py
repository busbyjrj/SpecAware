from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from aurora.fourier import FourierExpansion


class SentenceEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(SentenceEncoder, self).__init__()
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.get_embedding_dim(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

    def get_embedding_dim(self, model_name):
        if model_name == "sentence-transformers/all-MiniLM-L6-v2":
            return 384
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, sentences):
        with torch.no_grad():
            return self.model.encode(sentences, convert_to_tensor=True)


class AttributeEncoder(nn.Module):
    def __init__(self, embedding_dim=32, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(AttributeEncoder, self).__init__()
        self.sentence_encoder = SentenceEncoder(model_name)

        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(self.sentence_encoder.embedding_dim, embedding_dim)
        self.fc.apply(self.weight_init)

        self.processed_cache = {}
        self._preprocess_sentences()
        
        del self.sentence_encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _preprocess_sentences(self, sentences=None):
        if isinstance(sentences, str):
            sentences = [sentences]
        elif sentences is None:
            sentences = {"avc": "AVIRIS-Classic",
                         "avng": "AVIRIS-NG",
                         "av3": "AVIRIS-3",
                         "L1": "L1 Calibrated Radiance",
                         "L2": "L2 Surface Reflectance",
                         "others": "other HSI sensors",
                         }
                         
        for key, sentence in sentences.items():
            with torch.no_grad():
                embedding = self.sentence_encoder(sentence)
                self.processed_cache[key] = embedding.detach()

    def forward(self, sentence):
        if sentence in self.processed_cache:
            sentence_embedding = self.processed_cache[sentence].detach()
        else:
            raise ValueError(f"Sentence {sentence} not found in cache")

        sentence_embedding = sentence_embedding.to(self.fc.weight.device)
        sentence_embedding = self.fc(sentence_embedding)

        return sentence_embedding


class MLPResLayer(nn.Module):
    def __init__(self, in_dim=128, out_dim=128, hidden_dim=None):
        super(MLPResLayer, self).__init__()
        hidden_dim = in_dim * 2 if hidden_dim is None else hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if in_dim != out_dim:
            self.proj = nn.Linear(in_dim, out_dim)
            self.proj.apply(self.weight_init)
        else:
            self.proj = nn.Identity()

        self.mlp.apply(self.weight_init)
        
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.mlp(x) + self.proj(x)
        


class FWHMEncoder(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=64, min_fwhm=5.0, max_fwhm=15.0):
        super(FWHMEncoder, self).__init__()
        self.min_fwhm = min_fwhm
        self.max_fwhm = max_fwhm
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.mlp.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, fwhm):
        # 1. min-max normalize fwhm to [0, 1]
        normed = (fwhm - self.min_fwhm) / (self.max_fwhm - self.min_fwhm)
        # 2. encode fwhm to embedding
        normed = normed.unsqueeze(-1)  # [B, C] -> [B, C, 1]
        fwhm_embedding = self.mlp(normed)
        return fwhm_embedding


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


class SpectralEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SpectralEmbedding, self).__init__()
        # in this experiment, we just use the VNIR band and SWIR band from 350-2550 nm
        self.wavelength_encoder = FourierExpansion(350.0, 2550.0)
        self.fwhm_encoder = FWHMEncoder(embedding_dim)

        self.attribute_encoder = AttributeEncoder(embedding_dim//2)
        self.embedding_dim = embedding_dim
        
        self.fusion_weights = nn.Parameter(torch.tensor([0.8, 0.2]))
        
        self.spectral_enhancement_mlp = MLPResLayer(embedding_dim, embedding_dim)
        
        self.cross_modal_fusion = CrossModalFusion(embedding_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, wavelengths, fwhm, sensor_name, data_level):
        if isinstance(sensor_name, list):
            sensor_name = str(list(set(sensor_name))[0])
        if isinstance(data_level, list):
            data_level = str(list(set(data_level))[0])
            
        # spectral feature encoding
        wavelengths_embedding = self.wavelength_encoder(wavelengths, self.embedding_dim)
        fwhm_embedding = self.fwhm_encoder(fwhm)
        
        weights = F.softmax(self.fusion_weights, dim=0)
        spectral_embedding = (weights[0] * wavelengths_embedding + 
                            weights[1] * fwhm_embedding)  # [B, C, 128]
        spectral_embedding = self.spectral_enhancement_mlp(spectral_embedding)
        
        # attribute feature encoding
        sensor_name_embedding = self.attribute_encoder(sensor_name).repeat(
            wavelengths_embedding.shape[0], wavelengths_embedding.shape[1], 1)
        data_level_embedding = self.attribute_encoder(data_level).repeat(
            wavelengths_embedding.shape[0], wavelengths_embedding.shape[1], 1)
        sensor_embedding = torch.cat([sensor_name_embedding, data_level_embedding], dim=-1)
        
        # fusion
        final_embedding = self.cross_modal_fusion(spectral_embedding, sensor_embedding)
        final_embedding = final_embedding + spectral_embedding
        
        return final_embedding


class SpectralAwareTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, num_heads=4, num_layers=1):
        super().__init__()
        self.input_dim = input_dim

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            dim_feedforward=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=True,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, spectral_features):
        ### input: [B, C, wv_planes]
        output = self.transformer_encoder(spectral_features) + spectral_features
        
        return output
