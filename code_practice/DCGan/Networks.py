import torch
import torch.nn as nn

class Generator(nn.Module):
    """[생성기]
    Args:
        nz ([int]) : [입력 노이즈 크기]
        gis ([int]) : [출력 피쳐맵 너비]
        nc ([int]) : [출력 피쳐맵 RGB차원]
    """
    def __init__(self,nz=100,gis=64,nc=3):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(nz,gis*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(gis*8),
            nn.ReLU(True),
            # (gis*8, 4, 4)
            nn.ConvTranspose2d(gis*8,gis*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(gis*4),
            nn.ReLU(True),
            # (gis*4, 8, 8)
            nn.ConvTranspose2d(gis*4,gis*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(gis*2),
            nn.ReLU(True),
            # (gis*2, 16, 16)
            nn.ConvTranspose2d(gis*2,gis,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(gis),
            nn.ReLU(True),
            # (gis, 32, 32)
            nn.ConvTranspose2d(gis,nc,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
            # (nc, 64, 64)
        )
    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self,nc=3,dis=64,):
        super(Discriminator,self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(nc,dis,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(dis,dis*2,4,2,1,bias=False),
            nn.BatchNorm2d(dis*2),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(dis*2,dis*4,4,2,1,bias=False),
            nn.BatchNorm2d(dis),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(dis*4,dis*8,4,2,1,bias=False),
            nn.BatchNorm2d(dis*8),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(dis*8,1,4,1,0,bias=False),
            nn.sigmoid()
        )
