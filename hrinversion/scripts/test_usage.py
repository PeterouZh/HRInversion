


if __name__ == '__main__':
  import torch
  import torch.nn.functional as F
  from hrinversion import VGG16ConvLoss
  
  bs = 1
  # Note that the shortest side of the image must be larger than 32 pixels.
  img_size = 1024

  # Dummy data
  target = (torch.rand(bs, 3, img_size, img_size).cuda() - 0.5) * 2  # [-1, 1]
  pred = (torch.rand(bs, 3, img_size, img_size).cuda() - 0.5) * 2  # [-1, 1]
  pred.requires_grad_(True)
  
  # VGG conv-based perceptual loss
  percep_loss = VGG16ConvLoss().cuda().requires_grad_(False)

  # high-level perceptual loss: d_h
  # percep_loss = VGG16ConvLoss(fea_dict={'features_2': 0., 'features_7': 0., 'features_14': 0.,
  #                                       'features_21': 0.0002, 'features_28': 0.0005,
  #                                       }).cuda().requires_grad_(False)

  fea_target = percep_loss(target)
  fea_pred = percep_loss(pred)
  
  loss = F.mse_loss(fea_pred, fea_target, reduction='sum') / bs
  loss.backward()
  
  pass
