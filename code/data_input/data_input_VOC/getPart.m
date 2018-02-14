function getPart()
partNum=5;
partList={'bird','cat','cow','dog','horse','sheep'};
theConf=configurations;

for i=1:numel(partList)
    Name_batch=partList{i};
    objset=readAnnotation(Name_batch,theConf);

    tarSize=[224,224];
    getIMDBMask(Name_batch,objset,partNum,tarSize,theConf);
    tarSize=[227,227];
    getIMDBMask(Name_batch,objset,partNum,tarSize,theConf);
end
end


function getIMDBMask(Name_batch,objset,partNum,tarSize,theConf)
objNum=numel(objset);
mask=zeros(tarSize(1),tarSize(2),objNum,'uint8');
partPos=cell(objNum,1);
for i=1:objNum
    tmp=getPart_cate(objset(i),theConf,tarSize);
    mask(:,:,i)=uint8(tmp);
    pos=ones(2,partNum).*(-1);
    for part=1:partNum
        [hL,wL]=find(tmp==part);
        if(isempty(hL))
            continue;
        end
        pos(1,part)=mean(hL);
        pos(2,part)=mean(wL);
    end
    partPos{i}=pos;
    fprintf('%s   scale %d   %d\n',Name_batch,tarSize(1),i);
end
filename=sprintf('../results/imdb/%d/%s/partMask.mat',tarSize(1),Name_batch);
save(filename,'mask','partPos');
end


function mask=getPart_cate(obj,theConf,tarSize)
Batch_Name = obj.name;
switch(Batch_Name)
    case 'bird'
        partMerge={[1,2,3,4],[5,6,7,8],[9,10,11,12],13};
    case 'cat'
        partMerge={[1,2,3,4,5,6],[7,8],[9,10,11,12],[13,14,15,16],17};
    case 'cow'
        partMerge={[1,2,3,4,5,6,7,8],[9,10],[11,12,13,14],[15,16,17,18]};
    case 'dog'
        partMerge={[1,2,3,4,5,6,20],[7,8],[9,10,11,12],[13,14,15,16],17};
    case 'horse'
        partMerge={[1,2,3,4,5,6],[9,10],[11,12,13,14,30,31],[15,16,17,18,32,33]};
    case 'sheep'
        partMerge={[1,2,3,4,5,6,7,8],[9,10],[11,12,13,14],[15,16,17,18]};
    otherwise
        error('Errors here.');
end
[mask,bndbox]=getMask(obj,theConf,partMerge);
mask=mask(bndbox.ymin:bndbox.ymax,bndbox.xmin:bndbox.xmax);
yL=round(linspace(1,size(mask,1),tarSize(1)));
xL=round(linspace(1,size(mask,2),tarSize(2)));
mask=mask(yL,xL);
end


function [mask,bndbox]=getMask(obj,theConf,partMerge)
for i=1:numel(partMerge)
    pL=partMerge{i};
    for j=1:numel(pL)
        [tmp,bndbox]=getPart_one(obj,pL(j),theConf);
        if((i==1)&&(j==1))
            mask=zeros(size(tmp));
        end
        mask(tmp>0)=i;
    end
end
end


function [part_mask,bndbox]=getPart_one(obj, partID, theConf)

images_path = [theConf.data.imgdir, obj.folder];
anno_file = [theConf.data.annodir,'/%s.mat'];
imname = obj.filename;
batch_name = obj.name;
bndbox = obj.bndbox;
bndbox.ymax=str2double(bndbox.ymax);
bndbox.ymin=str2double(bndbox.ymin);
bndbox.xmax=str2double(bndbox.xmax);
bndbox.xmin=str2double(bndbox.xmin);
batch2classID=containers.Map('KeyType','char','ValueType','int32');
batch2classID('bird')=3;
batch2classID('cat')=8;
batch2classID('cow')=10;
batch2classID('dog')=12;
batch2classID('horse')=13;
batch2classID('sheep')=17;
classID=batch2classID(batch_name);

pimap = part2ind();
img = imread([images_path, '/', imname]);
% load annotation -- anno
load(sprintf(anno_file, imname(1:end-4)));

[cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);

tmp=(cls_mask==classID).*(part_mask==partID);
tmp([1:bndbox.ymin,bndbox.ymax:size(tmp,1)],:)=0;
tmp(:,[1:bndbox.xmin,bndbox.xmax:size(tmp,2)])=0;
inst_mask=inst_mask.*uint8(tmp);
part_mask=part_mask.*uint8(tmp);

for h=1:size(part_mask,1)
    for w=1:size(part_mask,2)
        if cls_mask(h,w)~=classID
            part_mask(h,w)=0;
            inst_mask(h,w)=0;
        elseif ((part_mask(h,w)~=partID) || (h>bndbox.ymax) || (h<bndbox.ymin) || (w<bndbox.xmin) || (w>bndbox.xmax))
            % skip if no part annotation or pixel outside of bndbox
            part_mask(h,w)=0;
            inst_mask(h,w)=0;
        end
    end
end
end


function [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap)
% Read the annotation and present it in terms of 3 segmentation mask maps (
% i.e., the class maks, instance maks and part mask). pimap defines a
% mapping between part name and index (See part2ind.m).

cls_mask = zeros(size(img,1), size(img,2), 'uint8');
inst_mask = zeros(size(img,1), size(img,2), 'uint8');
part_mask = zeros(size(img,1), size(img,2), 'uint8');

for oo = 1:numel(anno.objects)
    % The objects are ordered such that later objects can occlude previous
    % objects
    obj = anno.objects(oo);
    class_ind = obj.class_ind;
    silh = obj.mask;            % the silhouette mask of the object
    assert(size(silh,1) == size(img,1) && size(silh,2) == size(img,2));
    inst_mask(silh) = oo;
    cls_mask(silh) = class_ind;
    
    for pp = 1:numel(obj.parts)
        part_name = obj.parts(pp).part_name;
        assert(isKey(pimap{class_ind}, part_name));     % must define part index for every part
        assert(all(silh(obj.parts(pp).mask)));          % all part region is a subregion of the object
        pid = pimap{class_ind}(part_name);
        part_mask(obj.parts(pp).mask) = pid;
    end
end
end


function pimap = part2ind()
% Define the part index of each objects. 
% One can merge different parts by using the same index for the
% parts that are desired to be merged. 
% For example, one can merge 
% the left lower leg (llleg) and the left upper leg (luleg) of person by setting: 
% pimap{15}('llleg')      = 19;               % left lower leg
% pimap{15}('luleg')      = 19;               % left upper leg

pimap = cell(20, 1);                    
% Will define part index map for the 20 PASCAL VOC object classes in ascending
% alphabetical order (the standard PASCAL VOC order). 
for ii = 1:20
    pimap{ii} = containers.Map('KeyType','char','ValueType','int32');
end

% [aeroplane]
pimap{1}('body')        = 1;                
pimap{1}('stern')       = 2;                                      
pimap{1}('lwing')       = 3;                % left wing
pimap{1}('rwing')       = 4;                % right wing
pimap{1}('tail')        = 5;                
for ii = 1:10
    pimap{1}(sprintf('engine_%d', ii)) = 10+ii; % multiple engines
end
for ii = 1:10
    pimap{1}(sprintf('wheel_%d', ii)) = 20+ii;  % multiple wheels
end

% [bicycle]
pimap{2}('fwheel')      = 1;                % front wheel
pimap{2}('bwheel')      = 2;                % back wheel
pimap{2}('saddle')      = 3;               
pimap{2}('handlebar')   = 4;                % handle bar
pimap{2}('chainwheel')  = 5;                % chain wheel
for ii = 1:10
    pimap{2}(sprintf('headlight_%d', ii)) = 10 + ii;
end

% [bird]
pimap{3}('head')        = 1;
pimap{3}('leye')        = 2;                % left eye
pimap{3}('reye')        = 3;                % right eye
pimap{3}('beak')        = 4;               
pimap{3}('torso')       = 5;            
pimap{3}('neck')        = 6;
pimap{3}('lwing')       = 7;                % left wing
pimap{3}('rwing')       = 8;                % right wing
pimap{3}('lleg')        = 9;                % left leg
pimap{3}('lfoot')       = 10;               % left foot
pimap{3}('rleg')        = 11;               % right leg
pimap{3}('rfoot')       = 12;               % right foot
pimap{3}('tail')        = 13;

% [boat]
% only has silhouette mask 

% [bottle]
pimap{5}('cap')         = 1;
pimap{5}('body')        = 2;

% [bus]
pimap{6}('frontside')   = 1;
pimap{6}('leftside')    = 2;
pimap{6}('rightside')   = 3;
pimap{6}('backside')    = 4;
pimap{6}('roofside')    = 5;
pimap{6}('leftmirror')  = 6;
pimap{6}('rightmirror') = 7;
pimap{6}('fliplate')    = 8;                % front license plate
pimap{6}('bliplate')    = 9;                % back license plate
for ii = 1:10
    pimap{6}(sprintf('door_%d',ii)) = 10 + ii;
end
for ii = 1:10
    pimap{6}(sprintf('wheel_%d',ii)) = 20 + ii;
end
for ii = 1:10
    pimap{6}(sprintf('headlight_%d',ii)) = 30 + ii;
end
for ii = 1:20
    pimap{6}(sprintf('window_%d',ii)) = 40 + ii;
end

% [car]
keySet = keys(pimap{6});
valueSet = values(pimap{6});
pimap{7} = containers.Map(keySet, valueSet);         % car has the same set of parts with bus

% [cat]
pimap{8}('head')        = 1;
pimap{8}('leye')        = 2;                % left eye
pimap{8}('reye')        = 3;                % right eye
pimap{8}('lear')        = 4;                % left ear
pimap{8}('rear')        = 5;                % right ear
pimap{8}('nose')        = 6;
pimap{8}('torso')       = 7;   
pimap{8}('neck')        = 8;
pimap{8}('lfleg')       = 9;                % left front leg
pimap{8}('lfpa')        = 10;               % left front paw
pimap{8}('rfleg')       = 11;               % right front leg
pimap{8}('rfpa')        = 12;               % right front paw
pimap{8}('lbleg')       = 13;               % left back leg
pimap{8}('lbpa')        = 14;               % left back paw
pimap{8}('rbleg')       = 15;               % right back leg
pimap{8}('rbpa')        = 16;               % right back paw
pimap{8}('tail')        = 17;               

% [chair]
% only has sihouette mask 

% [cow]
pimap{10}('head')       = 1;
pimap{10}('leye')       = 2;                % left eye
pimap{10}('reye')       = 3;                % right eye
pimap{10}('lear')       = 4;                % left ear
pimap{10}('rear')       = 5;                % right ear
pimap{10}('muzzle')     = 6;
pimap{10}('lhorn')      = 7;                % left horn
pimap{10}('rhorn')      = 8;                % right horn
pimap{10}('torso')      = 9;            
pimap{10}('neck')       = 10;
pimap{10}('lfuleg')     = 11;               % left front upper leg
pimap{10}('lflleg')     = 12;               % left front lower leg
pimap{10}('rfuleg')     = 13;               % right front upper leg
pimap{10}('rflleg')     = 14;               % right front lower leg
pimap{10}('lbuleg')     = 15;               % left back upper leg
pimap{10}('lblleg')     = 16;               % left back lower leg
pimap{10}('rbuleg')     = 17;               % right back upper leg
pimap{10}('rblleg')     = 18;               % right back lower leg
pimap{10}('tail')       = 19;               

% [diningtable]
% only has silhouette mask 

% [dog]
keySet = keys(pimap{8});
valueSet = values(pimap{8});
pimap{12} = containers.Map(keySet, valueSet);         	% dog has the same set of parts with cat, 
                                            		% except for the additional
                                            		% muzzle
pimap{12}('muzzle')     = 20;

% [horse]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{13} = containers.Map(keySet, valueSet);        	% horse has the same set of parts with cow, 
                                                        % except it has hoof instead of horn
remove(pimap{13}, {'lhorn', 'rhorn'});
pimap{13}('lfho') = 30;
pimap{13}('rfho') = 31;
pimap{13}('lbho') = 32;
pimap{13}('rbho') = 33;

% [motorbike]
pimap{14}('fwheel')     = 1;
pimap{14}('bwheel')     = 2;
pimap{14}('handlebar')  = 3;
pimap{14}('saddle')     = 4;
for ii = 1:10
    pimap{14}(sprintf('headlight_%d', ii)) = 10 + ii;
end

% [person]
pimap{15}('head')       = 1;
pimap{15}('leye')       = 2;                    % left eye
pimap{15}('reye')       = 3;                    % right eye
pimap{15}('lear')       = 4;                    % left ear
pimap{15}('rear')       = 5;                    % right ear
pimap{15}('lebrow')     = 6;                    % left eyebrow    
pimap{15}('rebrow')     = 7;                    % right eyebrow
pimap{15}('nose')       = 8;                    
pimap{15}('mouth')      = 9;                    
pimap{15}('hair')       = 10;                   

pimap{15}('torso')      = 11;                   
pimap{15}('neck')       = 12;           
pimap{15}('llarm')      = 13;                   % left lower arm
pimap{15}('luarm')      = 14;                   % left upper arm
pimap{15}('lhand')      = 15;                   % left hand
pimap{15}('rlarm')      = 16;                   % right lower arm
pimap{15}('ruarm')      = 17;                   % right upper arm
pimap{15}('rhand')      = 18;                   % right hand

pimap{15}('llleg')      = 19;               	% left lower leg
pimap{15}('luleg')      = 20;               	% left upper leg
pimap{15}('lfoot')      = 21;               	% left foot
pimap{15}('rlleg')      = 22;               	% right lower leg
pimap{15}('ruleg')      = 23;               	% right upper leg
pimap{15}('rfoot')      = 24;               	% right foot

% [pottedplant]
pimap{16}('pot')        = 1;
pimap{16}('plant')      = 2;

% [sheep]
keySet = keys(pimap{10});
valueSet = values(pimap{10});
pimap{17} = containers.Map(keySet, valueSet);        % sheep has the same set of parts with cow

% [sofa]
% only has sihouette mask 

% [train]
pimap{19}('head')       = 1;
pimap{19}('hfrontside') = 2;                	% head front side                
pimap{19}('hleftside')  = 3;                	% head left side
pimap{19}('hrightside') = 4;                	% head right side
pimap{19}('hbackside')  = 5;                 	% head back side
pimap{19}('hroofside')  = 6;                	% head roof side

for ii = 1:10
    pimap{19}(sprintf('headlight_%d',ii)) = 10 + ii;
end

for ii = 1:10
    pimap{19}(sprintf('coach_%d',ii)) = 20 + ii;
end

for ii = 1:10
    pimap{19}(sprintf('cfrontside_%d', ii)) = 30 + ii;   % coach front side
end

for ii = 1:10
    pimap{19}(sprintf('cleftside_%d', ii)) = 40 + ii;   % coach left side
end

for ii = 1:10
    pimap{19}(sprintf('crightside_%d', ii)) = 50 + ii;  % coach right side
end

for ii = 1:10
    pimap{19}(sprintf('cbackside_%d', ii)) = 60 + ii;   % coach back side
end

for ii = 1:10
    pimap{19}(sprintf('croofside_%d', ii)) = 70 + ii;   % coach roof side
end


% [tvmonitor]
pimap{20}('screen')     = 1;
end
