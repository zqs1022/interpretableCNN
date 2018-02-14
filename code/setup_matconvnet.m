function setup_matconvnet()
try
    addpath(genpath('../matconvnet-1.0-beta24/'));
    vl_setupnn;
catch
    %% setup matconvnet
    system('wget -O ../matconvnet.zip --no-check-certificate https://github.com/vlfeat/matconvnet/archive/v1.0-beta24.zip');
    system('unzip ../matconvnet.zip -d ../');
    system('rm ../matconvnet.zip');


    %% compile matconvnet
    cd ../matconvnet-1.0-beta24/
    addpath(genpath('matlab'));
    vl_compilenn('enableGpu', true); %% Please check setting options in http://www.vlfeat.org/matconvnet/install/ before running this, so that you can revise the command according to your system.
    cd ../code
    addpath(genpath('../matconvnet-1.0-beta24/'));
    vl_setupnn;
end

%% download pre-trained CNNs
mkdir('../nets');
if(~exist('../nets/imagenet-vgg-verydeep-16.mat','file'))
    system('wget -O ../nets/imagenet-vgg-verydeep-16.mat --no-check-certificate http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat');
end
if(~exist('../nets/imagenet-vgg-m.mat','file'))
    system('wget -O ../nets/imagenet-vgg-m.mat --no-check-certificate http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat');
end
if(~exist('../nets/imagenet-vgg-s.mat','file'))
    system('wget -O ../nets/imagenet-vgg-s.mat --no-check-certificate http://www.vlfeat.org/matconvnet/models/imagenet-vgg-s.mat');
end
if(~exist('../nets/imagenet-caffe-alex.mat','file'))
    system('wget -O ../nets/imagenet-caffe-alex.mat --no-check-certificate http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat');
end
