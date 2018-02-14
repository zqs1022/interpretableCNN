function stability=computeStability(partRate,net,Name_batch,selectedPatternRatio,patchNumPerPattern,partList,theConf,layerID,convnet,meanData)

if(strcmp(Name_batch,'cub200'))
    objset=readAnnotation_sampling(Name_batch,theConf);
else
    objset=readAnnotation(Name_batch,theConf);
end

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
patNum=round(numel(net.layers{targetLayer}.weights{2})*partRate);
selectedPatternNum=round(patNum*selectedPatternRatio);
pos=zeros(2,patNum,imgNum);
score=zeros(patNum,imgNum);
isFlip=false;
for imgID=1:imgNum
    if(~validImg(imgID))
        continue;
    end
    [res,I,~]=getCNNFeature(objset(imgID),net,theConf,isFlip,meanData);
    x=double(res(targetLayer+1).x(:,:,1:patNum));
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
distSqrtVar=sort(distSqrtVar(isnan(distSqrtVar)==0),'ascend');
stability=mean(distSqrtVar(1:min(selectedPatternNum,numel(distSqrtVar))))/norm([ih,iw]);
end
