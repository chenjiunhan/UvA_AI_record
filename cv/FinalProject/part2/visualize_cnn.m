%LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.21
%run /home/jaqq/matconvnet/matlab/vl_setupnn;
net = load('data/pre_trained_model.mat');
net = net.net;
vl_simplenn_display(net) ;