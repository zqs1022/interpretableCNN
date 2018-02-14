function objset_neg=getNegObjSet(theConf,Name_batch)
MaxObjNum=1000;

objset_neg(MaxObjNum).folder=[];
objset_neg(MaxObjNum).filename=[];
objset_neg(MaxObjNum).name=[];
objset_neg(MaxObjNum).bndbox=[];
objset_neg(MaxObjNum).ID=[];
for i=1:MaxObjNum
    filename=sprintf('%05d',i);
    img=imread(['../data/neg/',filename,'.JPEG']);
    objset_neg(i).folder='../../neg';
    objset_neg(i).filename=[filename,'.JPEG'];
    objset_neg(i).name=[Name_batch,'_neg'];
    [h,w,~]=size(img);
    bndbox.xmin=1;
    bndbox.ymin=1;
    bndbox.xmax=w;
    bndbox.ymax=h;
    objset_neg(i).bndbox=bndbox;
    objset_neg(i).ID=1000000000+i;
end
