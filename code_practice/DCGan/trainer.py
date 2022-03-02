import Networks
import dataloader
import sub
import os
import argparse
from torch.backends import cudnn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import torchvision.utils as vutils

def main(config):
    
    # Directory
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    
    #GPU setting
    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    #Dataset setting
    transform=transforms.Compose([
                            transforms.Resize((config.image_size,config.image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),       
    ])
    dataset=dataloader.My_data(path=path+config.dataroot,transform=transform)
    iterator=torch.utils.data.DataLoader(dataset,
                                         batch_size=config.batch_size,
                                         shuffle=True,
                                         )
    
    #model setting
    G=Networks.Generator(nz=config.nz,gis=config.image_size)
    D=Networks.Discriminator(dis=config.image_size)
    
    G.to(device)
    D.to(device)
    
    G.apply(sub.weights_init)
    D.apply(sub.weights_init)
    
    # Optimizer setting
    optG=optim.Adam(G.parameters(),
                    lr=config.g_lr,
                    betas=(config.beta1,config.beta2))
    
    optD=optim.Adam(D.parameters(),
                    lr=config.d_lr,
                    betas=(config.beta1,config.beta2))
    
    sub.print_network(G,"Generator")
    sub.print_network(D,"Discriminator")
    
    # scheduler
    Gscheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optG, T_0=10, T_mult=1, eta_min=0.00001, last_epoch=-1)
    Dshceduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optD, T_0=10, T_mult=1, eta_min=0.00001, last_epoch=-1)
    
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)
    #Loss_function
    criterion=nn.BCELoss()
    img_list=[]
    G_losses=[]
    D_losses=[]
    iters=0
    print("훈련 시작")
    print(device)
    for epoch in tqdm(range(config.num_iterations)):
        for i,data in enumerate(iterator,0):
            G.train()
            D.train()
            
            D.zero_grad()
            data=data.to(device)
            #real,fake
            y_real=torch.Tensor(data.size(0)).fill_(1.0).to(device)
            y_fake=torch.Tensor(data.size(0)).fill_(0.0).to(device)
            z=torch.randn(data.size(0),config.nz,1,1,device=device)
            #D
            D_real=D(data).view(-1)
            errD_real=criterion(D_real,y_real)
            errD_real.backward()
            D_x=D_real.mean().item()
            
            G_fake=G(z)
            D_fake=D(G_fake.detach()).view(-1)
            errD_fake=criterion(D_fake,y_fake)
            errD_fake.backward()
            D_G_z1=D_fake.mean().item()
            
            errD=errD_fake+errD_real
            
            optD.step()
            
            #G
            G.zero_grad()
            output=D(G_fake).view(-1)
            errG=criterion(output,y_real)
            errG.backward()
            D_G_z2=output.mean().item()
            optG.step()
            
            if i % 50 ==0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, config.num_iterations , i, len(iterator),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            
            if (iters % 500 == 0) or ((epoch == config.num_iterations-1) and (i == len()-1)):
                with torch.no_grad():
                    fake = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1
    torch.save(G.state_dict(),path+config.model_save_dir+"/G.pth")
    torch.save(D.state_dict(),path+config.model_save_dir+"/D.pth")
    G=Networks.Generator(nz=config.nz,gis=config.image_size)
    G.load_state_dict(torch.load(path+config.model_save_dir+"/G.pth"))
    D=Networks.Discriminator(dis=config.image_size)
    D.load_state_dict(torch.load(path+config.model_save_dir+"/D.pth"))
        

if __name__=="__main__":
    path=os.getcwd()
    parser=argparse.ArgumentParser()
    # Data options
    parser.add_argument("--dataroot",type=str,default="/data/celeba/images",help="dataset 경로")
    parser.add_argument("--noise_kind",type=str,default=None,help="Gn이 거치는 noise 종류")
    parser.add_argument("--image_size",type=int,default=64,help="Resize할 이미지 크기")

    # Model options
    parser.add_argument("--g_repeat_num",type=int,default=6,help="G-residual block 의 반복 수")
    parser.add_argument("--d_repeat_num",type=int,default=6,help="D-residual block 의 반복 수")

    # train or test
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        
    # Trainig options
    parser.add_argument('--batch_size', type=int, default=512)    
    parser.add_argument('--g_lr', type=float, default=2e-4,)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.)
    parser.add_argument('--beta2', type=float, default=0.99)   
    parser.add_argument('--num_critic', type=int, default=1,help="G 훈련 대비 D 훈련 횟수") 
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument("--nz",type=int,default=100,help="입력 노이즈 크기")
    
    # Output options
    parser.add_argument('--visualize_interval', type=int, default=5000, help="시각화 간격")
    
    # Directories    
    parser.add_argument('--model_save_dir', type=str, default='/models',help="model의 가중치 저장 디렉터리")
    parser.add_argument('--model_load_dir',type=str,default='/models',help="사용 시에 가중치 저장되어 있는 디렉터리")
    config=parser.parse_args()
    print(path+config.dataroot)
    main(config) 