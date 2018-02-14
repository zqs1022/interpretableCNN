function imdb=getImdb_cub200(Name_batch,theConf,meta,IsTrain)
[objset,trainList]=readAnnotation(Name_batch,theConf);
objset_neg=getNegObjSet(theConf,Name_batch);

objset_neg=reshape(repmat(objset_neg,[4,1]),[1,4*numel(objset_neg)]); %%%%%%%%

num_pos=length(objset);
num_neg=length(objset_neg);

image_size=meta.normalization.imageSize(1:2);
data=zeros(image_size(1),image_size(2),3,num_pos,'single');
data_neg=zeros(image_size(1),image_size(2),3,num_neg,'single');
for i=1:num_pos
    tar=i;
    IsFlip=false;
    [I_patch,~]=getI(objset(i),theConf,image_size,IsFlip);
    data(:,:,:,tar)=I_patch;
end
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
total_images=num_pos+num_neg;
imdb.images.data=zeros(image_size(1),image_size(2),3,total_images,'single');
list=1:(num_pos);
imdb.images.data(:,:,:,list)=data;
imdb.images.data(:,:,:,setdiff(1:total_images,list))=data_neg;
clear data data_neg
imdb.images.labels=ones(1,total_images,'single').*(-1);
imdb.images.labels(1,list)=1;
imdb.images.set=zeros(1,total_images,'uint8');
trainList=reshape(trainList,[1,numel(trainList)]);
list_train=find(imdb.images.labels==-1);
tmp=1:round(num_neg*0.5);
list_train=reshape(list_train(tmp),[1,numel(tmp)]);
list_train=[trainList,list_train];
imdb.images.set(1:end)=2;
imdb.images.set(list_train)=1;
dataMean=mean(imdb.images.data(:,:,:,imdb.images.set==1),4);
imdb.images.data=bsxfun(@minus,imdb.images.data,dataMean);
if(IsTrain)
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
