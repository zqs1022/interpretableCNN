function stability=step_computeStability_multiClass(net,layerID,theConf,folder,partList)
selectPatternRatio=1.0;
patchNumPerPattern=100;
load(sprintf('./mat/%s/imdb.mat',folder),'meta');
Name_list=net.meta.classes.name;

convnet=getConvNetPara(net);
net=vl_simplenn_tidy(net);
net=our_vl_simplenn_move(net,'gpu') ;
assert(layerID>0);

NameNum=numel(Name_list);
for i=1:NameNum
    Name_batch=Name_list{i};
    if(strcmp(Name_batch,'n01443537'))
        partList=1;
    end
    [tmp,tmp_score]=computeStability_multiClass(net,Name_batch,selectPatternRatio,patchNumPerPattern,partList,theConf,layerID,convnet,meta.dataMean);
    if(i==1)
        stability=zeros(numel(tmp),NameNum);
        score=zeros(numel(tmp),NameNum);
    end
    stability(:,i)=tmp;
    score(:,i)=tmp_score;
end
for i=1:size(stability,1)
    [~,idx]=max(score(i,:));
    stability(i,1)=stability(i,idx);
end
stability=stability(:,1);
selectedPatternNum=round(selectPatternRatio*numel(stability));
stability=sort(stability(isnan(stability)==0),'ascend');
stability=mean(stability(1:min(selectedPatternNum,numel(stability))));
disp(stability);
end


function [stability_filter,score_filter]=computeStability_multiClass(net,Name_batch,selectedPatternRatio,patchNumPerPattern,partList,theConf,layerID,convnet,meanData)

objset=readAnnotation(Name_batch,theConf);

imgNum=length(objset);
partNum=numel(partList);
validImg=zeros(imgNum,1);
for i=1:partNum
    partID=partList(i);
    filename=sprintf('%s%s/truth_part%02d.mat',theConf.data.readCode,Name_batch,partID);
    a=load(filename);
    for img=1:imgNum
        if(~isempty(a.truth(img).pHW_center))
            validImg(img)=true;
        end
    end
end
targetLayer=convnet.targetLayers(layerID)-1;
patNum=numel(net.layers{targetLayer}.weights{2});
pos=zeros(2,patNum,imgNum);
score=zeros(patNum,imgNum);
isFlip=false;
for imgID=1:imgNum
    if(~validImg(imgID))
        continue;
    end
    [res,I,~]=getCNNFeature(objset(imgID),net,theConf,isFlip,meanData);
    x=double(res(targetLayer+1).x);
    xh=size(x,1);
    [v,idx]=max(x,[],1);
    [v,tmp]=max(v,[],2);
    tmp=reshape(tmp,[1,patNum]);
    idx_h=idx(tmp+(0:patNum-1).*xh);
    idx_w=reshape(tmp,[1,patNum]);
    theScore=reshape(v,[patNum,1]);
    thePos=x2p_([idx_h;idx_w],layerID,convnet);
    pos(:,:,imgID)=thePos;
    score(:,imgID)=theScore;
end
ih=size(I,1);
iw=size(I,2);
distSqrtVar=getDistSqrtVar(pos,score,patchNumPerPattern,partList,Name_batch,theConf);
stability_filter=distSqrtVar/norm([ih,iw]);
score_filter=mean(score,2);
end
