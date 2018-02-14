function setup_vocpart()
%% download images
if(~exist('../data/VOC_part','dir'))
    system('wget -O ../VOCPart.tar.gz --no-check-certificate http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz');
    mkdir('../data/VOC_part/')
    system('tar -xvzf ../VOCPart.tar.gz -C ../data/VOC_part/');
    system('rm ../VOCPart.tar.gz');
    system('wget -O ../img.tar http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar');
    system('tar -xvf ../img.tar -C ../data/VOC_part/');
    system('rm ../img.tar');
end


%% settings
conf.data.catedir='../data/VOC_part/';
conf.data.imgdir='../data/VOC_part/VOCdevkit/VOC2010/JPEGImages/';
conf.data.readCode='./data_input/data_input_VOC/';

conf.data.minArea=50^2;

conf.output.dir='./mat/';

Name_batch={'bird','cat','cow','dog','horse','sheep'};
for i=1:numel(Name_batch)
    mkdir([conf.output.dir,Name_batch{i}]);
    save([conf.output.dir,Name_batch{i},'/conf.mat'],'conf');
end

addpath(genpath('./tool'));
addpath(conf.data.readCode);
