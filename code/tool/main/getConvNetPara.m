function convnet=getConvNetPara(net)
convLayers=[];
for l=1:numel(net.layers)
    if(strcmp(net.layers{l}.type,'conv')||strcmp(net.layers{l}.type,'conv_aNet')||strcmp(net.layers{l}.type,'conv_mask'))
        convLayers(end+1)=l;
    end
end
len=length(convLayers);
convnet.targetLayers=convLayers+1;
convnet.targetScale=zeros(1,len);
convnet.targetStride=zeros(1,len);
convnet.targetCenter=zeros(1,len);
for i=1:len
    tarLay=convLayers(i);
    layer=net.layers{tarLay};
    pad=layer.pad(1);
    scale=size(layer.weights{1},1);
    stride=layer.stride(1);
    if(i==1)
        convnet.targetStride(i)=stride;
        convnet.targetScale(i)=scale;
        convnet.targetCenter(i)=(1+scale-pad*2)/2;
    else
        IsPool=false;
        poolStride=0;
        poolSize=0;
        poolPad=0;
        for j=convLayers(i-1)+1:tarLay-1
            if(strcmp(net.layers{j}.type,'pool'))
                IsPool=true;
                poolSize=net.layers{j}.pool(1);
                poolStride=net.layers{j}.stride(1);
                poolPad=net.layers{j}.pad(1);
            end
        end
        convnet.targetStride(i)=(1+IsPool*(poolStride-1))*stride*convnet.targetStride(i-1);
        convnet.targetScale(i)=convnet.targetScale(i-1)+IsPool*(poolSize-1)*convnet.targetStride(i-1)+convnet.targetStride(i)*(scale-1);
        if(IsPool)
            convnet.targetCenter(i)=(scale-pad*2-1)*poolStride*convnet.targetStride(i-1)/2+(convnet.targetCenter(i-1)+convnet.targetStride(i-1)*(poolSize-2*poolPad-1)/2);
        else
            convnet.targetCenter(i)=(scale-pad*2-1)*convnet.targetStride(i-1)/2+convnet.targetCenter(i-1);
        end
    end
end
end
