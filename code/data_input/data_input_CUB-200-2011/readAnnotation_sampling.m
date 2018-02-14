function objset=readAnnotation_sampling(Name_batch,theConf)
MaxObjNum=1000000;

fileID=fopen([theConf.data.catedir,'image_class_labels.txt'],'r');
idClassPair=textscan(fileID,'%d %d');
fclose(fileID);
if((length(Name_batch)==1)&&(Name_batch(1)-'0'>0)&&(Name_batch(1)-'0'<=9))
    logInd=idClassPair{2}==str2num(Name_batch);
    imgIds=idClassPair{1}(logInd);
else
    imgIds=idClassPair{1};
end

fileID=fopen([theConf.data.catedir,'train_test_split.txt'],'r');
idTrainTestPair=textscan(fileID,'%d %d');
fclose(fileID);
logInd=idTrainTestPair{2}==0;
testIds=idTrainTestPair{1}(logInd);
[~,idx,~]=intersect(imgIds,testIds);
imgIds=imgIds(idx);

fileID=fopen([theConf.data.catedir,'classes.txt'],'r');
batchClassnamePair=textscan(fileID,'%d %s');
fclose(fileID);
%logInd=batchClassnamePair{1}==str2num(Name_batch);
%classname=batchClassnamePair{2}(logInd);

fileID=fopen([theConf.data.catedir,'images.txt'],'r');
idNamePair=textscan(fileID,'%d %s');
fclose(fileID);
[~,idx,~]=intersect(idNamePair{1},imgIds);
imgnames=idNamePair{2}(idx);

fileID=fopen([theConf.data.catedir,'bounding_boxes.txt'],'r');
idBndboxPair=textscan(fileID,'%d %d %d %d %d');
fclose(fileID);
[~,idx,~]=intersect(idBndboxPair{1},imgIds);
x=idBndboxPair{2}(idx);
y=idBndboxPair{3}(idx);
width=idBndboxPair{4}(idx);
height=idBndboxPair{5}(idx);

clear logInd fileID idClassPair idNamePair idBndboxPair

objset(MaxObjNum).folder=[];
objset(MaxObjNum).filename=[];
objset(MaxObjNum).name=[];
objset(MaxObjNum).bndbox=[];
objset(MaxObjNum).ID=[];

j=0;
for i=1:length(imgnames)
    filename=imgnames(i);
    words=strsplit(filename{1},'.');
    classID=str2num(words{1});
    logInd=batchClassnamePair{1}==classID;
    classname=batchClassnamePair{2}(logInd);
    
    name=classname;
    bndbox.xmin=int2str(x(i));
    bndbox.xmax=int2str(x(i)+width(i));
    bndbox.ymin=int2str(y(i));
    bndbox.ymax=int2str(y(i)+height(i));

    j=j+1;
    objset(j).folder='.';
    objset(j).filename=filename{1};
    objset(j).name=name{1};
    objset(j).bndbox=bndbox;
    objset(j).ID=j;
    if(j>MaxObjNum)
        error('MaxObjNum is too small.');
    end
end
objset=objset(1:j);

load(['./data_input/data_input_CUB-200-2011/sampling.mat'],'validObj');
objset=objset(validObj);
for i=1:length(objset)
    objset(i).ID=i;
end
