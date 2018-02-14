function readPartTruth(Name_batch, partID, theConf)
MaxObjNum=100000;
minArea=theConf.data.minArea;

fileID=fopen([theConf.data.catedir,'image_class_labels.txt'],'r');
idClassPair=textscan(fileID,'%d %d');
fclose(fileID);
if ~strcmp(Name_batch, '0')
    logInd=idClassPair{2}==str2num(Name_batch);
    imgIds=idClassPair{1}(logInd);
else
    imgIds=[1:length(idClassPair{1})];
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
% logInd=batchClassnamePair{1}==str2num(Name_batch);
% classname=batchClassnamePair{2}(logInd);

fileID=fopen([theConf.data.catedir,'parts/part_locs.txt'],'r');
idPartlocPair=textscan(fileID,'%d %d %d %d %d');
fclose(fileID);
logInd=idPartlocPair{2}==partID;
idPartlocPair{1}=idPartlocPair{1}(logInd);
idPartlocPair{3}=idPartlocPair{3}(logInd);
idPartlocPair{4}=idPartlocPair{4}(logInd);
idPartlocPair{5}=idPartlocPair{5}(logInd);
[~,idx,~]=intersect(idPartlocPair{1},imgIds);
imgIds=idPartlocPair{1}(idx);
partX=double(idPartlocPair{3}(idx));
partY=double(idPartlocPair{4}(idx));
partVisible=idPartlocPair{5}(idx);

fileID=fopen([theConf.data.catedir,'images.txt'],'r');
idNamePair=textscan(fileID,'%d %s');
fclose(fileID);
[~,idx,~]=intersect(idNamePair{1},imgIds);
imgnames=idNamePair{2}(idx);

fileID=fopen([theConf.data.catedir,'bounding_boxes.txt'],'r');
idBndboxPair=textscan(fileID,'%d %d %d %d %d');
fclose(fileID);
[~,idx,~]=intersect(idBndboxPair{1},imgIds);
x=double(idBndboxPair{2}(idx));
y=double(idBndboxPair{3}(idx));
width=double(idBndboxPair{4}(idx));
height=double(idBndboxPair{5}(idx));

partset(MaxObjNum).pHW_center=[];
partset(MaxObjNum).obj=[];
j=0;
for i=1:length(imgnames)
    if (partVisible(i)==1)
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
        if(~IsAreaValid(bndbox,minArea))
            continue;
        end
        j=j+1;
        obj.folder='.'; %classname;
        obj.filename=filename{1};
        obj.name=name{1};
        obj.bndbox=bndbox;
        obj.ID=j;
        partset(j).obj=obj;
        partset(j).pHW_center=[(partY(i)-y(i))/height(i)*224;(partX(i)-x(i))/width(i)*224];

        if(j>MaxObjNum)
            error('MaxObjNum is too small.');
        end
    else
        j=j+1;
    end
end
partset=partset(1:j);
clear truth
truth=partset;

load([theConf.output.dir,Name_batch,'/sampling.mat'],'validObj');
truth=truth(validObj);
for i=1:length(truth)
    if ~isempty(truth(i).obj)
        truth(i).obj.ID=i;
    end
end

save(sprintf('%s%s/truth_part%02d.mat',theConf.output.dir,Name_batch,partID),'truth');