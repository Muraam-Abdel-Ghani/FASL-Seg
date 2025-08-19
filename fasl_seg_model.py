import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel
from llf_decoder import LLFDecoder

class FASLSegModel(nn.Module):
    def __init__(self, num_classes):
        super(FASLSegModel, self).__init__()
        self.segformer = SegformerModel.from_pretrained( 'nvidia/mit-b5', output_hidden_states=True)
        
        # Projections for skip connections
        self.llf_decoder = LLFDecoder(num_classes)
        
        # -------- Compress channels and increase feature map size for LLF output
        self.conv1LLF = nn.Conv2d(128, 64, kernel_size=1) # [batch, 128,128,128] -> [batch,64,128,128]
        self.bn_conv1LLF = nn.BatchNorm2d(64)
        self.leakyReLU_conv1LLF = nn.LeakyReLU()
        
        #-----------new code: changed output channels 32-> 64
        self.conv2LLF = nn.Conv2d(64,64,kernel_size=1) # [batch,64,128,128] -> [batch, 64,128,128]
        self.bn_conv2LLF = nn.BatchNorm2d(64)
        self.leakyReLU_conv2LLF = nn.LeakyReLU()
        
        # -- QUESTION: Kernel=1 and padding=0  OR kernel=3 and padding=1
        self.finalConv = nn.Conv2d(64, num_classes, kernel_size=3, padding=1) # conv after concat to compress channels [batch,64,512,512] -> [batch,num_classes,512,512]
        
    def forward(self, x):
        # Get encoder hidden states from SegFormer (skips are in hidden_states)
        encoder_outputs = self.segformer(x)
        
        llfeatures = self.llf_decoder(encoder_outputs) # -> [batch, 128,128,128]

        
        # --- Preparing mask from Output of LLF Decoder ---
        # print(f'LLF_mask bef conv after crosscom: {llf_cross_comm_output.shape}')
        llfeatures = self.conv1LLF(llfeatures) # [batch, 128,128,128] -> [batch,64,128,128]     
        llfeatures = self.bn_conv1LLF(llfeatures)
        llfeatures = self.leakyReLU_conv1LLF(llfeatures)
        
        # print(f'LLF_mask conv1: {llf_mask.shape}')
        
        llfeatures = self.conv2LLF(llfeatures) # [batch,64,128,128] -> [batch, 64,128,128]
        llfeatures = self.bn_conv2LLF(llfeatures)
        llfeatures = self.leakyReLU_conv2LLF(llfeatures)
        
        llfeatures = F.interpolate(llfeatures, size=(256, 256), mode='bilinear', align_corners=False) # [batch,64,128,128] -> [batch,64,256,256]

        transposed_mask = F.interpolate(llfeatures, size=(512, 512), mode='bilinear', align_corners=False) #[batch,64,256,256] -> [batch,64,512,512]

        
        final_mask = self.finalConv(transposed_mask) # [batch, 64 ,512,512] -> [batch, num_classes,512,512]

        return final_mask # [batch, num_classes, 512,512]
    