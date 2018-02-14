function stability=step_computeStability(partRate,net,layerID,theConf,fileRoot,partList)
selectPatternRatio=1.0;
patchNumPerPattern=100;
load(sprintf('%s/imdb.mat',fileRoot),'meta');
Name_batch=net.meta.classes.name{1};

if(strcmp(Name_batch,'n01443537'))
    partList=1;
end

convnet=getConvNetPara(net);
net=vl_simplenn_tidy(net);
net=our_vl_simplenn_move(net,'gpu') ;
assert(layerID>0);
stability=computeStability(partRate,net,Name_batch,selectPatternRatio,patchNumPerPattern,partList,theConf,layerID,convnet,meta.dataMean);
disp(stability);
end
