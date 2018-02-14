function net = vgg_vd_16(labelNum,lossWeight,opts)
channel_num=512;
mag=0.1;
netname='vgg-vd-16';

partRate=1;
textureRate=0;
output_num=labelNum;
net=load('../nets/imagenet-vgg-verydeep-16.mat');

net.layers=net.layers(1:end-7);
net.layers{end-1}.learningRate=[1,5];
net=add_block_mask(net,lossWeight, opts,'6',3,3,512,channel_num,1,1,partRate,textureRate,mag); %%%%%%%%%%%%%%%%%
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net=add_block_mask(net,lossWeight,opts,'7',7,7,channel_num,4096,1,0,partRate,textureRate,mag); %%%%%%%%%%%%%%%%%

net = add_dropout(net, opts, 'fc--7') ;
net = add_block(net, opts, 'fc--8', 1, 1, 4096, 4096, 1, 0) ;
net = add_dropout(net, opts, 'fc--8') ;

net = add_block(net, opts, '9', 1, 1, 4096, output_num, 1, 0) ;
net.layers(end) = [] ;
net.meta.classes.name={'pos'};
net.meta.classes.description={'pos'};
net.meta.normalization.imageSize=net.meta.normalization.imageSize(1:3);
net.meta.netname=netname;
end
