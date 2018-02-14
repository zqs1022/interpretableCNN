function objset=readAnnotation(Name_batch,theConf)
MaxObjNum=5000;

minArea=theConf.data.minArea;
objset(MaxObjNum).folder=[];
objset(MaxObjNum).filename=[];
objset(MaxObjNum).name=[];
objset(MaxObjNum).bndbox=[];
objset(MaxObjNum).ID=[];
load([theConf.data.imgdir,Name_batch,'_obj/img/data.mat'],'samples');
root=[theConf.data.imgdir,Name_batch,'_obj/img/img/'];
files=dir(root);
for i=1:length(files)-2
    folder=[Name_batch,'_obj/img/img/'];
    filename=sprintf('%05d.jpg',i);
    xmin=samples(i).obj.bndbox.xmin;
    ymin=samples(i).obj.bndbox.ymin;
    xmax=samples(i).obj.bndbox.xmax;
    ymax=samples(i).obj.bndbox.ymax;
    bndbox.xmin=int2str(xmin);
    bndbox.xmax=int2str(xmax);
    bndbox.ymin=int2str(ymin);
    bndbox.ymax=int2str(ymax);
    if(~IsAreaValid(bndbox,minArea))
        continue;
    end
    objset(i).folder=folder;
    objset(i).filename=filename;
    objset(i).name=filename;
    objset(i).bndbox=bndbox;
    objset(i).ID=i;
end
objset=objset(1:length(files)-2);
end


function pd=IsAreaValid(bndbox,minArea)
xmin=str2double(bndbox.xmin);
xmax=str2double(bndbox.xmax);
ymin=str2double(bndbox.ymin);
ymax=str2double(bndbox.ymax);
if((xmax-xmin+1)*(ymax-ymin+1)>=minArea)
    pd=true;
else
    pd=false;
end
end
