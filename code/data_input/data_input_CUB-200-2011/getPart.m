function getPart()
partMerge={1,2,3,4,5,6,7,8,9,10,1,12,13,14,15};
Name_batch='cub200';
theConf=configurations;

partNum=numel(partMerge);
objset=readAnnotation(Name_batch,theConf);
fp=fopen([theConf.data.catedir,'parts/part_locs.txt'],'r');
matrix=fscanf(fp,'%f%f%f%f%f\n');
fclose(fp);
matrix=reshape(matrix,[5,size(matrix,1)/5]);
partInfo=matrix(1:4,:);
partInfo(2,:)=0;
for i=1:numel(partMerge)
    list=partMerge{i};
    for j=1:numel(list)
        partInfo(2,matrix(2,:)==list(j))=i;
    end
end
clear matrix
partInfo=partInfo(:,partInfo(2,:)>0);
tarSize=[224,224];
getIMDBMask(Name_batch,objset,partNum,partInfo,tarSize);
tarSize=[227,227];
getIMDBMask(Name_batch,objset,partNum,partInfo,tarSize);
end


function getIMDBMask(Name_batch,objset,partNum,partInfo,tarSize)
objNum=numel(objset);
partPos=cell(objNum,1);
for i=1:objNum
    bndbox = objset(i).bndbox;
    bndbox.ymax=str2double(bndbox.ymax);
    bndbox.ymin=str2double(bndbox.ymin);
    bndbox.xmax=str2double(bndbox.xmax);
    bndbox.xmin=str2double(bndbox.xmin);
    pos=ones(2,partNum).*(-1);
    valid=[];
    for part=1:partNum
        list=find((partInfo(1,:)==i).*(partInfo(2,:)==part)>0);
        if(isempty(list))
            continue;
        end
        theP=partInfo([4,3],list);
        theP=theP(:,(theP(1,:)>=bndbox.ymin).*(theP(1,:)<=bndbox.ymax).*(theP(2,:)>=bndbox.xmin).*(theP(2,:)<=bndbox.xmax)>0);
        if(isempty(theP))
            continue;
        end
        pos(:,part)=mean(theP,2);
        valid(end+1)=part;
    end
    if(~isempty(valid))
        pos(1,valid)=(pos(1,valid)-bndbox.ymin)./(bndbox.ymax-bndbox.ymin).*(tarSize(1)-1)+1;
        pos(2,valid)=(pos(2,valid)-bndbox.xmin)./(bndbox.xmax-bndbox.xmin).*(tarSize(2)-1)+1;
    end
    partPos{i}=pos;
    fprintf('%s   scale %d   %d\n',Name_batch,tarSize(1),i);
end
filename=sprintf('../results/imdb/%d/%s/partMask.mat',tarSize(1),Name_batch);
save(filename,'partPos');
end
