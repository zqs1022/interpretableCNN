function step_drawRF(theConf,Name_batch,layerID,epochNum)
showNum=10;
integrateEnergy=0.003; %0.005;
visualizationStep=50;

rng(1);

fileRoot='./draw_RF';
mkdir(fileRoot);
createImages(Name_batch,layerID,fileRoot,epochNum,showNum,integrateEnergy,theConf,visualizationStep);
end


function createImages(nameList,layerID,fileRoot,epochNum,showNum,integrateEnergy,theConf,visualizationStep)
num=numel(nameList);
for i=1:num
    filename=sprintf('./%s/%s/net-epoch-%d.mat',theConf.output.dir,nameList{i},epochNum);
    net=load(filename);
    theConf.output.dir=['./',fileRoot,'/'];
    theConf.data.Name_batch=nameList{i};
    generateAMTOriginalImages(net.net,layerID,showNum,integrateEnergy,theConf,visualizationStep);
    drawRawFMaskF(net.net,layerID,showNum,theConf,visualizationStep);
    computeRF(net.net,layerID,showNum,theConf,visualizationStep);
    drawRF(net.net,layerID,theConf,visualizationStep);
end
end


function generateAMTOriginalImages(net,layerID,showNum,integrateEnergy,theConf,visualizationStep)
batchSize=32;
samplingTimes=1000;

objset=readAnnotation(theConf.data.Name_batch,theConf);
if(strcmp(theConf.data.Name_batch,'cub200'))
    objset=objset(1:500);
end
imgNum=numel(objset);
root_ori=fullfile(theConf.output.dir,'img_ori');
mkdir(root_ori);
convnet=getConvNetPara(net);
net=vl_simplenn_tidy(net);
net=our_vl_simplenn_move(net,'gpu');
assert(layerID>0);
labelNum=size(net.layers{end}.class,3);
ih=net.meta.normalization.imageSize(1);
iw=net.meta.normalization.imageSize(2);
isFlip=zeros(1,imgNum);
targetLayer=convnet.targetLayers(layerID)-1;
upperLayer=convnet.targetLayers(layerID+1)-1;
a=load(sprintf('./mat/%s/imdb.mat',theConf.data.Name_batch),'meta');
for img=1:batchSize:imgNum
    imgList=img:min(img+batchSize-1,imgNum);
    batS=numel(imgList);
    net.layers{end}.class=gpuArray(ones(1,1,labelNum,batS));
    images=getImg(imgList,theConf,[ih,iw],objset,a.meta);
    res=our_vl_simplenn(net,gpuArray(images));
    clear images
    x=res(targetLayer+1).x;
    x=maskX(x,net,upperLayer);
    if(img==1)
        depth=size(x,3);
        depStat=zeros(depth,samplingTimes*imgNum,'single');
        depMax=zeros(depth,imgNum,'single');
    end
    for dep=1:depth
        tmp=x(:,:,dep,:);
        tmplen=numel(tmp);
        if(tmplen>0)
            tmp=reshape(tmp(randperm(tmplen)),[1,tmplen]);
            inputlen=numel(imgList)*samplingTimes;
            tmp=repmat(tmp,[1,ceil(inputlen/tmplen)]);
            depStat(dep,(img-1)*samplingTimes+(1:inputlen))=tmp(1:inputlen);
        end
    end
    depMax(:,imgList)=reshape(max(max(x,[],1),[],2),[depth,batS]);
end
[depStat,~]=sort(depStat,2,'descend');
validMap=zeros(depth,imgNum);
tarP=round(integrateEnergy*size(depStat,2));
tau=depStat(:,tarP);
clear depStat
for dep=1:depth
    validList=find(depMax(dep,:)>=tau(dep));
    validNum=numel(validList);
    if(validNum<showNum)
        [tttmp,validList]=sort(depMax(dep,:),'descend');
        validList=sort(validList(1:showNum));
        validNum=showNum;
        tau(dep)=tttmp(showNum);
    end
    validMap(dep,validList)=true;
    assert(validNum>=showNum);
    tmp=randperm(validNum);
    validMap(dep,validList(tmp(showNum+1:end)))=false;    
end
count=zeros(depth,1);
validDepthVisualization=1:visualizationStep:depth;
imgList=1:imgNum;
for dep=1:depth
    if(ismember(dep,validDepthVisualization))
        validList=find(validMap(dep,imgList)>0);
        validNum=numel(validList);
        for id=1:validNum
            count(dep)=count(dep)+1;
            imgID=imgList(validList(id));
            obj=objset(imgID);
            isF=isFlip(imgID);
            Iori=getI(obj,theConf,[ih,iw],isF);
            filename=sprintf('%s/%s-%04d-%05d.jpg',root_ori,theConf.data.Name_batch,dep,count(dep));
            imwrite(uint8(Iori),filename);
        end
    end
end
filename=sprintf('%s/%s_tau.mat',root_ori,theConf.data.Name_batch);
save(filename,'tau');
end


function computeRF(net,layerID,showNum,theConf,visualizationStep)
batchSize=64;
stride=9; %6; %3;
occludeSize=33; %22; %11;
topOccNum=4; %15;

root_ori=fullfile(theConf.output.dir,'img_ori');
convnet=getConvNetPara(net);
net=vl_simplenn_tidy(net);
net=our_vl_simplenn_move(net,'gpu');
assert(layerID>0);
ih=net.meta.normalization.imageSize(1);
iw=net.meta.normalization.imageSize(2);
targetLayer=convnet.targetLayers(layerID)-1;
depth=size(net.layers{targetLayer}.weights{1},4);
occludeNum=ceil((ih-occludeSize+1)/stride)*ceil((iw-occludeSize+1)/stride);
occludeList=repmat(struct('sh',[],'sw',[]),[1,occludeNum]);
c=0;
for sh=1:stride:ih-occludeSize+1
    for sw=1:stride:iw-occludeSize+1
        c=c+1;
        occludeList(c).sh=sh;
        occludeList(c).sw=sw;
    end
end
a=load(sprintf('./mat/%s/imdb.mat',theConf.data.Name_batch),'meta');
for dep=1:visualizationStep:depth
    filename_out=sprintf('%s/%s-%04d_template.mat',root_ori,theConf.data.Name_batch,dep);
    if(exist(filename_out,'file'))
        disp(filename_out);
        continue;
    end
    template=zeros(ih*2,iw*2,showNum);
    for img=1:showNum
        filename=sprintf('%s/%s-%04d-%05d.jpg',root_ori,theConf.data.Name_batch,dep,img);
        I=imread(filename);
        [oriValue,~,m]=getValue(net,single(I)-single(a.meta.dataMean),dep,convnet,layerID);
        if(img==1)
            map=zeros(size(m,1),size(m,2),showNum);
        end
        map(:,:,img)=m;
        tarValue=zeros(1,occludeNum);
        pHW=zeros(2,occludeNum);
        for occStart=1:batchSize:occludeNum
            occList=occStart:min(occStart+batchSize,occludeNum);
            images=getImg_occlude(occludeList(occList),I,occludeSize,a.meta);
            [tarValue(occList),pHW(:,occList),~]=getValue(net,images,dep,convnet,layerID);
            clear images
        end
        [v,idx]=sort(tarValue,'ascend');
        for i=1:topOccNum
            id=idx(i);
            val=max(oriValue-v(i),0);
            hmin=occludeList(id).sh-pHW(1,id);
            hmax=occludeList(id).sh+occludeSize-1-pHW(1,id);
            wmin=occludeList(id).sw-pHW(2,id);
            wmax=occludeList(id).sw+occludeSize-1-pHW(2,id);
            hmin=max(round(hmin+ih),1);
            hmax=min(round(hmax+ih),ih*2);
            wmin=max(round(wmin+iw),1);
            wmax=min(round(wmax+iw),iw*2);
            template(hmin:hmax,wmin:wmax,img)=template(hmin:hmax,wmin:wmax,img)+val;
        end
    end
    save(filename_out,'template','map');
    disp(filename_out);
    clear template map
end
end


function drawRF(net,layerID,theConf,visualizationStep)
drawThreshold=0.5;

root_ori=fullfile(theConf.output.dir,'img_ori');
root_RF=fullfile(theConf.output.dir,'img_RF');
mkdir(root_RF);
filename=sprintf('%s/%s_tau.mat',root_ori,theConf.data.Name_batch);
tau=load(filename,'tau');
convnet=getConvNetPara(net);
assert(layerID>0);
ih=net.meta.normalization.imageSize(1);
iw=net.meta.normalization.imageSize(2);
targetLayer=convnet.targetLayers(layerID)-1;
depth=size(net.layers{targetLayer}.weights{1},4);
for dep=1:visualizationStep:depth
    filename=sprintf('%s/%s-%04d_template.mat',root_ori,theConf.data.Name_batch,dep);
    a=load(filename,'template','map');
    [xh,xw,showNum]=size(a.map);
    RFMask=getRFMask(a,ih,iw,xh,xw,layerID,convnet);
    for img=1:showNum
        filename=sprintf('%s/%s-%04d-%05d.jpg',root_ori,theConf.data.Name_batch,dep,img);
        I=imread(filename);
        assert((size(I,1)==ih)&&(size(I,2)==iw));
        map=a.map(:,:,img);
        theTau=tau.tau(dep);
        wei=getRegion(ih,iw,xh,xw,map,theTau,drawThreshold,RFMask);
        I_RF=drawEdge(wei,I);
        filename=sprintf('%s/%s-%04d-%05d.jpg',root_RF,theConf.data.Name_batch,dep,img);
        imwrite(uint8(I_RF),filename);
    end
end
end


function RFMask=getRFMask(a,ih,iw,xh,xw,layerID,convnet)
T=sum(a.template,3);
Gauss=fspecial('gaussian',50,30);
T=imfilter(T,Gauss,'replicate','same');
RFMask=zeros(ih,iw,xh*xw,'single');
c=0;
for W=1:xw
    for H=1:xh
        c=c+1;
        thePos=x2p_([H;W],layerID,convnet);
        hmin=ih-thePos(1)+1;
        wmin=iw-thePos(2)+1;
        RFMask(:,:,c)=T(hmin:hmin+ih-1,wmin:wmin+ih-1);
    end
end
RFMask=gpuArray(RFMask);
end


function region=getRegion(ih,iw,xh,xw,map,tau,drawThreshold,RFMask)
map=reshape(map,[1,1,xh*xw]);
tau=min(tau,max(map(:)));
map(map<tau)=0;
map=min(map.*(1.5/max(max(map(:)),0.00000000001)),1);
map(map>=tau)=1;
map=repmat(map,[ih,iw,1]);
wei=max(RFMask.*map,[],3);
region=single(gather(wei>=drawThreshold*max(wei(:))));
end


function [tarValue,pHW,map]=getValue(net,images,dep,convnet,layerID)
batchS=size(images,4);
net.layers{end}.class=gpuArray(ones(1,1,size(net.layers{end}.class,3),batchS));
res=our_vl_simplenn(net,gpuArray(images));
targetLayer=convnet.targetLayers(layerID)-1;
upperLayer=convnet.targetLayers(layerID+1)-1;
x=res(targetLayer+1).x;
x=maskX(x,net,upperLayer);
[xh,xw,~,~]=size(x);
xx=reshape(x(:,:,dep,:),[xh*xw,batchS]);
[~,idx]=max(xx,[],1);
pw=reshape(ceil(idx./xh),[1,batchS]);
ph=reshape(idx-(pw-1).*xh,[1,batchS]);
pHW=x2p_([ph;pw],layerID,convnet);
tarValue=reshape(sum(sum(x(:,:,dep,:),1),2),[batchS,1]);
map=reshape(gather(x(:,:,dep,:)),[xh,xw,batchS]);
end


function images=getImg_occlude(occludeList,I,occludeSize,meta)
batS=numel(occludeList);
images=repmat(single(I)-single(meta.dataMean),[1,1,1,batS]);
for i=1:batS
    hList=occludeList(i).sh:occludeList(i).sh+occludeSize-1;
    wList=occludeList(i).sw:occludeList(i).sw+occludeSize-1;
    images(hList,wList,:,i)=0;
end
end


function images=getImg(imgList,theConf,tarSize,objset,meta)
batS=numel(imgList);
ih=tarSize(1);
iw=tarSize(2);
images=zeros(ih,iw,3,batS,'single');
isFlip=false;
for i=1:batS
    I=getI(objset(imgList(i)),theConf,tarSize,isFlip);
    images(:,:,:,i)=single(I);
end
images=images-repmat(meta.dataMean,[1,1,1,batS]);
end


function [x,mask]=maskX(x,net,upperLayer)
x=max(x,0);
if(~strcmp(net.layers{upperLayer}.type,'conv_mask'))
    x=gather(x);
    return;
end
[h,w,depth,batchS]=size(x);
%[mu_x,mu_y,~]=getMu(x);
[mu_x,mu_y]=getMu_max(x);
posTempX=gpuArray(single(repmat(linspace(-1,1,h)',[1,w,depth,batchS])));
posTempY=gpuArray(single(repmat(linspace(-1,1,w),[h,1,depth,batchS])));
mask=abs(posTempX-repmat(mu_x,[h,w,1,1]));
mask=mask+abs(posTempY-repmat(mu_y,[h,w,1,1]));
mask=max(1-mask.*repmat(reshape(net.layers{upperLayer}.weights{3},[1,1,depth]),[h,w,1,batchS]),0);
mask(:,:,net.layers{upperLayer}.filter~=1,:)=1;
%assert(isempty(find(net.layers{upperLayer}.filter~=1)));
x=gather(x.*mask);
end


function [mu_x,mu_y]=getMu_max(x)
[h,w,depth,batchS]=size(x);
x=reshape(x,[h*w,depth,batchS]);
[~,p]=max(x,[],1);
p=reshape(p,[1,1,depth,batchS]);
mu_y=ceil(p./h);
mu_x=p-(mu_y-1).*h;
tmp=linspace(-1,1,h);
mu_x=reshape(tmp(mu_x),[1,1,depth,batchS]);
mu_y=reshape(tmp(mu_y),[1,1,depth,batchS]);
end


function drawRawFMaskF(net,layerID,showNum,theConf,visualizationStep)
root_ori=fullfile(theConf.output.dir,'img_ori');
root_maskF=fullfile(theConf.output.dir,'img_maskF');
mkdir(root_maskF);
convnet=getConvNetPara(net);
net=vl_simplenn_tidy(net);
net=our_vl_simplenn_move(net,'gpu');
assert(layerID>0);
ih=net.meta.normalization.imageSize(1);
targetLayer=convnet.targetLayers(layerID)-1;
depth=size(net.layers{targetLayer}.weights{1},4);
a=load(sprintf('./mat/%s/imdb.mat',theConf.data.Name_batch),'meta');
for dep=1:visualizationStep:depth
    for img=1:showNum
        filename=sprintf('%s/%s-%04d-%05d.jpg',root_ori,theConf.data.Name_batch,dep,img);
        I=imread(filename);
        [raw_x,mask,mask_x]=getMaskMap(net,single(I)-single(a.meta.dataMean),dep,convnet,layerID);
        filename=sprintf('%s/%s-%04d-%05d_rawX.jpg',root_maskF,theConf.data.Name_batch,dep,img);
        imagewrite(raw_x,filename);
        filename=sprintf('%s/%s-%04d-%05d_mask.jpg',root_maskF,theConf.data.Name_batch,dep,img);
        imagewrite(mask,filename);
        filename=sprintf('%s/%s-%04d-%05d_maskX.jpg',root_maskF,theConf.data.Name_batch,dep,img);
        imagewrite(mask_x,filename);
    end
    fprintf('%s-%04d.jpg\n',theConf.data.Name_batch,dep)
end
end


function [raw_x,mask,mask_x]=getMaskMap(net,images,dep,convnet,layerID)
net.layers{end}.class=gpuArray(ones(1,1,size(net.layers{end}.class,3),1));
res=our_vl_simplenn(net,gpuArray(images));
targetLayer=convnet.targetLayers(layerID)-1;
upperLayer=convnet.targetLayers(layerID+1)-1;
raw_x=res(targetLayer+1).x;
[mask_x,mask]=maskX(raw_x,net,upperLayer);
raw_x=gather(raw_x(:,:,dep));
mask=gather(mask(:,:,dep));
mask_x=gather(mask_x(:,:,dep));
end


function imagewrite(x,filename)
scaleW=16;

[xh,xw]=size(x);
x=uint8(x./max(x(:)).*255);
x=reshape(repmat(reshape(x,[1,xh,xw]),[scaleW,1,1]),[xh*scaleW,xw]);
x=reshape(repmat(reshape(x',[1,xw,xh*scaleW]),[scaleW,1,1]),[xw*scaleW,xh*scaleW]);
x=x';
imwrite(x,filename);
end
