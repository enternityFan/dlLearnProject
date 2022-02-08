# @Time: 16:55
# @Author:Phalange
# @File:diffScript.py
# @Software:PyCharm
# C’est la vie,enjoy it! :D
from IPython import display
from mxnet import np,npx
from d2l import mxnet as d2l

npx.set_np()
def f(x):
    return 3 * x ** 2 - 4 * x

def f2(x):
    return x **3 - 1/x


def numerical_lim(f,x,h):
    return (f(x+h) - f(x)) / h
def use_svg_display(): #@save
    """ 使用svg格式显示绘图"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5,2.5)): #@save
    """设置matplotlib的图标大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

#@save
def set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#@save
def plot(X,Y=None,xlabel=None,ylabel=None,legend=None,xlim=None,ylim=None,xscale='linear',yscale='linear',
         fmts=('-','m--','g-.','r:'),figsize=(3.5,2.5),axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X,"ndim") and X.ndim == 1 or isinstance(X,list) and not hasattr(X[0],"__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X,Y = [[]] * len(X),X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) !=len(Y):
        X = X * len(Y)
    axes.cla()
    for x,y,fmt in zip(X,Y,fmts):
        if len(x):
            axes.plot(x,y,fmt)
        else:
            axes.plot(y,fmt)

    set_axes(axes,xlabel,ylabel,xlim,ylim,xscale,yscale,legend)
    d2l.plt.show()


h = 0.1
for i in range(5):
    print(f'h={h:.5f},numerical limit={numerical_lim(f,1,h):.5f}')
    h *=0.1

x = np.arange(0,3,0.1)

plot(x,[f2(x),4*x-4],'x','f(x)',legend=['f(x)','Tangent line (x=1)'])
