function showResults(Name_batch,isMultiClassClassification,model,dataset)
if(isMultiClassClassification)
    load(['./mat/',Name_batch{1},'/conf.mat'],'conf');
else
    load(['./mat/',Name_batch,'/conf.mat'],'conf');
end
addpath(genpath('./tool'));
addpath(genpath('./tool/edges-master/piotr_toolbox/external'));
toolboxCompile;


%% compute classification error and location stability
fileRoot='./mat';
switch(model)
    case {'alexnet','vgg-m','vgg-s'}
        layerID=6;
    case 'vgg-vd-16'
        layerID=14;
    otherwise
        error('invalid model name.');
end
switch(dataset)
    case 'cub200'
        partList=[1,6,14];
    case 'vocpart'
        partList=[1,2,3];
    case 'ilsvrcanimalpart'
        partList=[1,2];
end
if(isMultiClassClassification)
    epochNum=80;
    [binaryerror,stability]=getResult_multiClass(Name_batch,layerID,fileRoot,epochNum,partList,conf);
else
    epochNum=50;
    [binaryerror,stability]=getResult({Name_batch},layerID,fileRoot,epochNum,partList,conf);
end
fprintf('binary error %f     location stability %f\n',binaryerror,stability)


%% draw image-resolution receptive field of neural activations of different filters.
if(~isMultiClassClassification)
    step_drawRF(conf,{Name_batch},layerID,epochNum);
end
end


function [binaryerror,stability]=getResult(nameList,layerID,fileRoot,epochNum,partList,conf)
partRate=1;
num=numel(nameList);
binaryerror=zeros(num,1);
stability=zeros(num,1);
for i=1:num
    filename=sprintf('./%s/%s/net-epoch-%d.mat',fileRoot,nameList{i},epochNum);
    try
        net=load(filename);
        conf.output.dir=['./',fileRoot,'/'];
        conf.data.Name_batch=nameList{i};
        imdbRoot=sprintf('./mat/%s',nameList{i});
        if(nargin<5)
            stability(i)=step_computeStability(partRate,net.net,layerID,conf,imdbRoot);
        else
            stability(i)=step_computeStability(partRate,net.net,layerID,conf,imdbRoot,partList);
        end
        binaryerror(i)=net.stats.val(end).binerr;
        clear net
    catch
        continue;
    end
end
end


function [binaryerror,stability]=getResult_multiClass(nameList,layerID,fileRoot,epochNum,partList,conf)
str=[];
if(numel(nameList)>10)
    for i=1:numel(nameList)
        str=sprintf('%s_%s',str,nameList{i}(end-1:end));
    end
else
    for i=1:numel(nameList)
        str=sprintf('%s_%s',str,nameList{i}(1:end));
    end
end
filename=sprintf('./%s/%s/net-epoch-%d.mat',fileRoot,str,epochNum);
net=load(filename);
conf.output.dir=['./',fileRoot,'/',str,'/'];
conf.data.Name_batch=nameList;
stability=step_computeStability_multiClass(net.net,layerID,conf,str,partList);
binaryerror=net.stats.val(end).binerr;
end
