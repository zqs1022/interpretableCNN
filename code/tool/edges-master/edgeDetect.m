function E=edgeDetect(I)
global edgemodel;
if(isempty(edgemodel))
    edgemodel=load('./tool/edges-master/model.mat');
end
E=edgesDetect(I,edgemodel);
end
