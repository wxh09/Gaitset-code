import scipy.io as sio
data = sio.loadmat('/home1/wxh/Gaitset-code/CCR.mat')
ccr = data['CCR']
nm = ccr[0,:,:,0]
bg = ccr[1,:,:,0]
cl = ccr[2,:,:,0]
a=1