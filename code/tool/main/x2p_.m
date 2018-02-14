function pHW=x2p_(xHW,layer,convnet)
Stride=convnet.targetStride(layer);
centerStart=convnet.targetCenter(layer);
pHW=centerStart+(xHW-1).*Stride;
end
