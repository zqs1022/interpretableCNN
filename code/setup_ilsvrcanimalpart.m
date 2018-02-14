function setup_ilsvrcanimalpart()
%% download images
if(~exist('../data/detanimalpart','dir'))
    system('rm -r ../data/detanimalpart');
    system('git clone https://github.com/zqs1022/detanimalpart.git');
    system('mv detanimalpart ../data/');
end


%% settings
conf.data.imgdir='../data/detanimalpart/';
conf.data.readCode='./data_input/data_input_DET-2015-subset-ILSVRC2013train/';
conf.data.minArea=50^2;

conf.output.dir='./mat/';

Name_batch={'n01443537','n01503061','n01639765','n01662784','n01674464','n01882714','n01982650','n02084071','n02118333','n02121808','n02129165','n02129604','n02131653','n02324045','n02342885','n02355227','n02374451','n02391049','n02395003','n02398521','n02402425','n02411705','n02419796','n02437136','n02444819','n02454379','n02484322','n02503517','n02509815','n02510455'};
for i=1:numel(Name_batch)
    mkdir([conf.output.dir,Name_batch{i}]);
    save([conf.output.dir,Name_batch{i},'/conf.mat'],'conf');
end

addpath(genpath('./tool'));
addpath(conf.data.readCode);
