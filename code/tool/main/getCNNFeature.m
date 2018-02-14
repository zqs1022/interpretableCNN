function [res,I_patch,I]=getCNNFeature(obj,theNet,theConf,IsFlip,meanData)
theNet.layers{end}.class=theNet.layers{end}.class(:,:,:,1);
theNet.layers{end}.class(:)=1;
[I_patch,I]=getI(obj,theConf,theNet.meta.normalization.imageSize(1:2),IsFlip);
im_=I_patch-meanData;
if(isa(theNet.layers{1}.weights{1},'gpuArray'))
    res=our_vl_simplenn(theNet,gpuArray(im_));
    for i=1:length(res)
        res(i).x=gather(res(i).x);
    end
else
    res=our_vl_simplenn(theNet,im_);
end
I_patch=uint8(I_patch);
end
