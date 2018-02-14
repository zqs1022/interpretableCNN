function [I_patch,I]=getI(obj,theConf,tarSize,IsFlip)
filename=[theConf.data.imgdir,obj.folder,'/',obj.filename];
try
    I=imread(filename);
catch
    disp(filename);
    error('cannot read the above image.');
end
[h,w,d]=size(I);
if(d==1)
    I=repmat(reshape(I,[h,w,1]),[1,1,3]);
end
xmin=max(str2double(obj.bndbox.xmin),1);
xmax=min(str2double(obj.bndbox.xmax),w);
ymin=max(str2double(obj.bndbox.ymin),1);
ymax=min(str2double(obj.bndbox.ymax),h);
I_patch=I(ymin:ymax,xmin:xmax,:);
I_patch=single(I_patch); % note: 0-255 range
I_patch=imresize(I_patch,tarSize,'bilinear');
if(IsFlip)
    I_patch=I_patch(:,end:-1:1,:);
    I=I(:,end:-1:1,:);
end
