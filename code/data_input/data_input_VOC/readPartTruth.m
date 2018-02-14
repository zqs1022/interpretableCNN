function readPartTruth(Name_batch, partID, theConf)
MaxObjNum=10000;
minArea=theConf.data.minArea;

annotationfile=[theConf.data.catedir,'truth_',Name_batch];
load(annotationfile,'truth');

partset(MaxObjNum).pHW_center=[];
partset(MaxObjNum).pHW_scale=[];
partset(MaxObjNum).obj=[];

j=0;
for i=1:length(truth)
    bndbox.xmin=int2str(truth(i).obj.bndbox.Wmin);
    bndbox.xmax=int2str(truth(i).obj.bndbox.Wmax);
    bndbox.ymin=int2str(truth(i).obj.bndbox.Hmin);
    bndbox.ymax=int2str(truth(i).obj.bndbox.Hmax);
    if(~IsAreaValid(bndbox,minArea))
        continue;
    end
    
    j=j+1;
    if isempty(truth(i).partbox(partID).Wmin)
        continue;
    end
    
    obj.folder=truth(i).obj.folder;
    obj.filename=truth(i).obj.filename;
    obj.name=truth(i).obj.name;
    obj.bndbox=bndbox;
    obj.ID=i;
    partset(j).obj=obj;
    
    width=double(truth(i).partbox(partID).Wmax-truth(i).partbox(partID).Wmin);
    height=double(truth(i).partbox(partID).Hmax-truth(i).partbox(partID).Hmin);
    x=double(truth(i).partbox(partID).Wmax+truth(i).partbox(partID).Wmin)/2-double(truth(i).obj.bndbox.Wmin);
    y=double(truth(i).partbox(partID).Hmax+truth(i).partbox(partID).Hmin)/2-double(truth(i).obj.bndbox.Hmin);

    partset(j).pHW_center=[y/(truth(i).obj.bndbox.Hmax-truth(i).obj.bndbox.Hmin)*224;x/(truth(i).obj.bndbox.Wmax-truth(i).obj.bndbox.Wmin)*224];
    partset(j).pHW_scale=[height/(truth(i).obj.bndbox.Hmax-truth(i).obj.bndbox.Hmin)*224;width/(truth(i).obj.bndbox.Wmax-truth(i).obj.bndbox.Wmin)*224];
end        
partset=partset(1:j);
clear truth
truth=partset;
save(sprintf('%s%s/truth_part%02d.mat',theConf.output.dir,Name_batch,partID),'truth');
