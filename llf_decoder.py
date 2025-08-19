import torch
import torch.nn as nn
import torch.nn.functional as F
# Tool Decoder
class LLFDecoder(nn.Module):
    def __init__(self, num_classes):
        super(LLFDecoder, self).__init__()
        
        embed_dim=128
        
        # Projections for skip connections
        self.proj1 = nn.Conv2d(64, embed_dim, kernel_size=1)   # Stage 1 projection (downscale 64 -> 128) [batch,64,128,128] -> [batch,128,128,128]
        self.bn_proj1 = nn.BatchNorm2d(embed_dim)
        self.leakyReLU_proj1 = nn.LeakyReLU()
        
        self.proj2 = nn.Conv2d(128, embed_dim, kernel_size=1)  # Stage 2 projection [batch,128,64,64] -> [batch,128,64,64]
        self.bn_proj2 = nn.BatchNorm2d(embed_dim)
        self.leakyReLU_proj2 = nn.LeakyReLU()
        
        self.proj3 = nn.Conv2d(320, embed_dim, kernel_size=1)  # Stage 3 projection [batch,320,32,32] -> [batch,128,32,32]
        self.bn_proj3 = nn.BatchNorm2d(embed_dim)
        self.leakyReLU_proj3 = nn.LeakyReLU()
        
        #gradual drop in channel size for fourth projection
        self.proj4_1 = nn.Conv2d(512, embed_dim*2, kernel_size=1)  # Stage 4 projection 1 [batch,512,16,16] -> [batch,256,16,16]
        self.bn_proj4_1 = nn.BatchNorm2d(embed_dim*2)
        self.leakyReLU_proj4_1 = nn.LeakyReLU()
        
        self.proj4_2 = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1)  # Stage 4 projection 2 [batch,256,16,16] -> [batch,128,16,16]
        self.bn_proj4_2 = nn.BatchNorm2d(embed_dim)
        self.leakyReLU_proj4_2 = nn.LeakyReLU()

        self.attention1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2, batch_first=True) # should output [batch,128,128,128]  (wih expansion [batch, 128, 256, 256])
        self.attention2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True) # should output [batch,128,64,64]  (with expansion [batch, 128, 256,256])
        
        
        # Reduce channel size of concatenated skip connections before returning for cross communication
        self.projSkips_1 = nn.Conv2d(512, embed_dim*2, kernel_size=1) # [batch, 512,128,128] -> [batch, 256,128,128]
        self.bn_projSkips_1 = nn.BatchNorm2d(embed_dim*2)
        self.leakyReLU_projSkips_1 = nn.LeakyReLU()
        
        self.projSkips_2 = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=1) # [batch, 256,128,128] -> [batch, 128,128,128]
        self.bn_projSkips_2 = nn.BatchNorm2d(embed_dim)
        self.leakyReLU_projSkips_2 = nn.LeakyReLU()
     

    def forward(self, encoder_outputs):
        # encoder_outputs come from segformer model (d2_segformer_model.py)
        hidden_states = encoder_outputs.hidden_states  # List of feature maps at different stages

        # Apply projections to each skip connection
        x1 = self.proj1(hidden_states[0])  # First stage [batch,64,128,128] -> [batch,128,128,128]
        x1 = self.bn_proj1(x1)
        x1 = self.leakyReLU_proj1(x1)
        
        x2 = self.proj2(hidden_states[1])  # Second stage [batch,128,64,64] -> [batch,128,64,64]
        x2 = self.bn_proj2(x2)
        x2 = self.leakyReLU_proj2(x2)
        
        x3 = self.proj3(hidden_states[2])  # Third stage [batch,320,32,32] -> [batch,128,32,32]
        x3 = self.bn_proj3(x3)
        x3 = self.leakyReLU_proj3(x3)
        
        
        x4 = self.proj4_1(hidden_states[3])  # Fourth stage (largest) [batch,512,16,16] -> [batch,256,16,16]
        x4 = self.bn_proj4_1(x4)
        x4 = self.leakyReLU_proj4_1(x4)
        
        x4 = self.proj4_2(x4)  # [batch,256,16,16] -> [batch,128,16,16]
        x4 = self.bn_proj4_2(x4)
        x4 = self.leakyReLU_proj4_2(x4)
        
        
    
        qX1 = x1.view(x1.size(0), x1.size(1), -1) # [batch, 128, 128x128] ([batch,128,256x256])
        qX1 = qX1.permute(0,2,1) # [batch, 128x128, 128]  ([batch,256x256,128])
        att_x1, _ = self.attention1(qX1,qX1,qX1) # since self-attention, query, key and value are same
        att_x1 = att_x1.permute(0, 2, 1).contiguous() # [batch, 128x128, 128] -> [batch, 128, 128x128] ([batch,256x256, 128]->[batch,128,256x256])
        att_x1 = att_x1.view(x1.size(0), x1.size(1), *x1.size()[2:])  # ******************[batch,128,128,128] ([batch,128,256,256])
        # print(att_x1.shape)
        
        qX2 = x2.view(x2.size(0), x2.size(1), -1) # [batch, 128, 128x128] ([batch,128,256x256])
        qX2 = qX2.permute(0,2,1) # [batch, 128x128, 128]  ([batch,256x256,128])
        att_x2, _ = self.attention2(qX2,qX2,qX2) # [batch, 128x128, 128] ([batch,256x256,128])
        att_x2 = att_x2.permute(0, 2, 1).contiguous() # [batch, 128x128, 128] -> [batch, 128, 128x128] ([batch,128,256x256])
        att_x2 = att_x2.view(x2.size(0), x2.size(1), *x2.size()[2:])  # ******************[batch,128,128,128] ([batch,128,256,256])
        
        att_x3 = x3  # [batch, 128,32,32]
        att_x4 = x4  # [batch, 128,16,16]


        x2_up = F.interpolate(att_x2, size=(128, 128), mode='bilinear', align_corners=False) # [batch,64,64,64] -> [batch,64,128,128]
     
        x3_up = F.interpolate(att_x3, size=(64, 64), mode='bilinear', align_corners=False) # [b,128,32,32] -> [b, 128,64,64]
        x3_up = F.interpolate(x3_up, size=(128, 128), mode='bilinear', align_corners=False) # [batch,64,64,64] -> [batch,64,128,128]
    
        x4_up = F.interpolate(att_x4, size=(32, 32), mode='bilinear', align_corners=False) # [b,128,16,16] -> [b, 128,32,32]
        x4_up = F.interpolate(x4_up, size=(64, 64), mode='bilinear', align_corners=False) # [b,128,32,32] -> [b, 128,64,64]
        x4_up = F.interpolate(x4_up, size=(128, 128), mode='bilinear', align_corners=False) # [batch,64,64,64] -> [batch,64,128,128]
  
        concatenated_skips = torch.cat([att_x1 , x2_up, x3_up, x4_up], dim=1) # 4x[b, 128,128,128] -> [b, 512,128,128] (4x[b,128,256,256]->[b,512,256,256])
        
      
        concatenated_skips = self.projSkips_1(concatenated_skips) # [b, 512,128,128] -> [b, 256,128,128] --required for cross-comm.
        concatenated_skips = self.bn_projSkips_1(concatenated_skips)
        concatenated_skips = self.leakyReLU_projSkips_1(concatenated_skips)

        
        concatenated_skips = self.projSkips_2(concatenated_skips) # [b, 256,128,128] -> [b, 128,128,128] --required for cross-comm.
        concatenated_skips = self.bn_projSkips_2(concatenated_skips)
        concatenated_skips = self.leakyReLU_projSkips_2(concatenated_skips)
        
        
        return concatenated_skips
        
        