


if __name__ == '__main__':
  import torch
  import torch.nn.functional as F
  from hrinversion import VGG16ConvLoss
  
  # img_size = 1024
  img_size = 64
  
  input = (torch.rand(1, 3, img_size, img_size).cuda() - 0.5) * 2  # [-1, 1]
  input = torch.nn.Parameter(input)
  target = (torch.rand(1, 3, img_size, img_size).cuda() - 0.5) * 2  # [-1, 1]
  percep_loss = VGG16ConvLoss().cuda()
  
  fea_input = percep_loss(input)
  fea_target = percep_loss(target)
  loss = F.mse_loss(fea_input, fea_target, reduction='sum')
  loss.backward()
  
  pass
