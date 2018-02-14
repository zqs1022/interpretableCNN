function net=initWeight(net,opts)
num=numel(net.layers);
for layer=1:num
    if(strcmp(net.layers{layer}.type,'conv'))
        [h,w,in,out]=size(net.layers{layer}.weights{1});
        net.layers{layer}.weights={init_weight(opts, h, w, in, out, 'single'),ones(out, 1, 'single')*opts.initBias};
    end
end
end
