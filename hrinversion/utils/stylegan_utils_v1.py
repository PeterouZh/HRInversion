import collections

import torch


@torch.no_grad()
def w_to_dict(w, ):
  w_dict = collections.OrderedDict()

  w_list = w.clone().split(2, dim=1)
  for idx, sub_w in enumerate(w_list):
    res = 2 ** (idx + 2)
    w_dict[f"b{res}.w"] = sub_w

  return w_dict


@torch.no_grad()
def w_swapping(content_w,
               style_w,
               gamma_style,
               swapped_layers):
  content_w_dict = w_to_dict(content_w)
  style_w_dict = w_to_dict(style_w)

  start_names = []
  for name in swapped_layers:
    start_names.append(f"b{name}.")
  start_names = tuple(start_names)

  style_params = style_w_dict
  swapped_params = content_w_dict
  for param_name, param in swapped_params.items():
    if param_name.startswith(start_names):
      style_param = style_params[param_name]
      param.data.copy_(param.data * (1 - gamma_style) + style_param.data * gamma_style)

  w_list = list(swapped_params.values())
  w_tensor = torch.cat(w_list, dim=1)
  return w_tensor


@torch.no_grad()
def layer_swapping(swapped_net,
                   style_net,
                   gamma_style,
                   swapped_layers):

  start_names = []
  for name in swapped_layers:
    start_names.append(f"b{name}.")
  start_names = tuple(start_names)

  style_params = style_net.state_dict()
  swapped_params = swapped_net.state_dict()
  for param_name, param in swapped_params.items():
    if param_name.startswith(start_names):
      style_param = style_params[param_name]
      param.data.copy_(param.data * (1 - gamma_style) + style_param.data * gamma_style)
  pass


@torch.no_grad()
def replace_ns(G,
               ns):
  ns_original = {}
  for name, buf in G.synthesis.named_buffers():
    if name in ns:
      ns_original[name] = buf.clone()
      buf.data.copy_(ns[name].data)
  return ns_original


@torch.no_grad()
def G_w(G,
        w,
        synthesis_kwargs=None,
        noise_mode='const',
        device='cuda'):
  """
  noise_mode: ['random', 'const', 'none']
  """
  if synthesis_kwargs is None:
    synthesis_kwargs = {
      'noise_mode': noise_mode
    }

  if not isinstance(w, torch.Tensor):
    w = torch.from_numpy(w).to(device)
  w = w.squeeze().unsqueeze(dim=0)

  img = G.synthesis(w, **synthesis_kwargs)
  return img


@torch.no_grad()
def G_w_ns(G,
           w,
           ns,
           synthesis_kwargs=None,
           ):

  ns_original = replace_ns(G, ns)

  img = G_w(G, w, synthesis_kwargs=synthesis_kwargs)

  replace_ns(G, ns_original)

  return img