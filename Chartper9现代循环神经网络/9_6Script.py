# @Time:2022-02-12 16:59
# @Author:Phalange
# @File:9_6Script.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D

from torch import nn

#@save
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)

    def forward(self,X,*args):
        raise NotImplementedError


#@save
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)

    def init_state(self,enc_outputs,*args):
        raise NotImplementedError

    def forward(self,X,state):
        raise NotImplementedError

#@save
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,enc_X,dec_X,*args):

        enc_outputs = self.encoder(enc_X,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)

