function [net,info]=learn_icnn(model,Name_batch,dropoutRate)
load(['./mat/',Name_batch,'/conf.mat'],'conf');
conf.data.Name_batch=Name_batch;
opts.dataDir=conf.data.imgdir;
opts.expDir=fullfile(conf.output.dir,conf.data.Name_batch);
opts.imdbPath=fullfile(opts.expDir,'imdb.mat');
opts.whitenData=true;
opts.contrastNormalization=true;
opts.networkType='simplenn';
try
    gpuDevice();
    opts.train=struct('gpus',1);
catch
    error('Errors here: GPU invalid.\n')
end

%% Prepare model
labelNum=1;
net=network_init(labelNum,model,dropoutRate,'networkType',opts.networkType);


%% Prepare data
if exist(opts.imdbPath,'file')
  imdb=load(opts.imdbPath) ;
else
  IsTrain=true;
  if(strcmp(Name_batch,'cub200'))
      imdb=getImdb_cub200(conf.data.Name_batch,conf,net.meta,IsTrain);
  else
      imdb=getImdb(conf.data.Name_batch,conf,net.meta,IsTrain);
  end
  mkdir(opts.expDir);
  save(opts.imdbPath,'-struct','imdb');
end

net.meta.classes.name=imdb.meta.classes(:)';


%% Train
[net,info]=our_cnn_train(net,imdb,getBatch(opts),'expDir',opts.expDir,net.meta.trainOpts,opts.train,'val',find(imdb.images.set==2));
end


function fn = getBatch(opts)
bopts = struct('numGpus', numel(opts.train.gpus)) ;
fn = @(x,y) getSimpleBatch(bopts,x,y) ;
end


function [images,labels]=getSimpleBatch(opts, imdb, batch)
images = imdb.images.data(:,:,:,batch) ;
labels = reshape(imdb.images.labels(:,batch),[1,1,size(imdb.images.labels,1),numel(batch)]);
if opts.numGpus > 0
  images = gpuArray(images) ;
end
end
