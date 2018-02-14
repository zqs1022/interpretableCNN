function imdb=getImdb(Name_batch,theConf,meta,IsTrain)
trainRate=0.9;
posRate=0.75;

objset=readAnnotation(Name_batch,theConf);
objset_neg=getNegObjSet(theConf,Name_batch);
num_pos=length(objset);
num_neg=length(objset_neg);
tarN=round(posRate/(1-posRate)*num_neg);
repN=ceil(tarN/(num_pos*2));
list_train_pos=round(linspace(1,num_pos,round(num_pos*trainRate)));
list_train_pos=[list_train_pos.*2-1,list_train_pos.*2];
if(repN>1)
    list_train_pos=repmat(list_train_pos,[repN,1])+repmat((0:repN-1)'.*(num_pos*2),[1,numel(list_train_pos)]);
    list_train_pos=reshape(sort(list_train_pos(:)),[1,numel(list_train_pos)]);
end
list_train_pos=list_train_pos(list_train_pos<=tarN);
list_train_neg=round(linspace(1,num_neg,round(num_neg*trainRate)));
list_train=sort([list_train_pos,list_train_neg+tarN]);

image_size=meta.normalization.imageSize(1:2);
data=zeros(image_size(1),image_size(2),3,num_pos*2,'single');
data_neg=zeros(image_size(1),image_size(2),3,num_neg,'single');
for i=1:num_pos
    tar=i*2-1;
    IsFlip=false;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    data(:,:,:,tar)=I_patch;
    tar=i*2;
    IsFlip=true;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    data(:,:,:,tar)=I_patch;
end
data=repmat(data,[1,1,1,repN]);
data=data(:,:,:,1:tarN);
theConf_neg=theConf;
if(isfield(theConf_neg.data,'imgdir_neg'))
    theConf_neg.data.imgdir=theConf_neg.data.imgdir_neg;
end
for i=1:num_neg
    tar=i;
    IsFlip=false;
    [I_patch,~]=getI(objset_neg(i),theConf_neg,image_size,IsFlip);
    data_neg(:,:,:,tar)=I_patch;
end
total_images=tarN+num_neg;
imdb.images.data=zeros(image_size(1),image_size(2),3,total_images,'single');
list=1:tarN;
imdb.images.data(:,:,:,list)=data;
imdb.images.data(:,:,:,tarN+(1:num_neg))=data_neg;
clear data data_neg
imdb.images.labels=ones(1,total_images,'single').*(-1);
imdb.images.labels(1,list)=1;
imdb.images.set=zeros(1,total_images,'uint8');
imdb.images.set(1:end)=2;
imdb.images.set(list_train)=1;
dataMean=mean(imdb.images.data(:,:,:,imdb.images.set==1),4);
imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean);
if(IsTrain)
    rng(0);
    list=randperm(total_images);
    imdb.images.data=imdb.images.data(:,:,:,list);
    imdb.images.labels=imdb.images.labels(:,list);
    imdb.images.set=imdb.images.set(:,list);
    [~,imdb.images.order]=sort(list);
end
imdb.meta.classes={Name_batch};
imdb.meta.sets={'train','val','test'} ;
imdb.meta.dataMean=dataMean;
end
