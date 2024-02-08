import flax.linen as nn
from type_shorthands import *

class Autoencoder(nn.Module):
  def setup(self):
    """Setup method to define submodules"""
    raise NotImplementedError("setup not implemented")
  
  def encode(self, x: R_bxdxdxc):
    """method to encode image into a representation"""
    raise NotImplementedError("encode not implemented")
  
  def decode(self, x: R_bxdxdxc):
    """method to decode image into a representation"""
    raise NotImplementedError("decode not implemented")