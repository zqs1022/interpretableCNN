function res=our_vl_simplenn(net, x, dzdy, res, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% About variable names in this code
%%% Variables in this code (e.g., alpha, beta, alpha_logZ_XXX, and mask)
%%% do not exactly represent concepts/notation with the same names in the paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
if(sum(x(:)<-10)==0)
    error('Errors in input');
end

opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false ;
opts.parameterServer = [] ;
opts.holdOn = false ;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;
assert(opts.backPropDepth > 0, 'Invalid `backPropDepth` value (!>0)');
backPropLim = max(n - opts.backPropDepth + 1, 1);

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
  if opts.skipForward
    error('simplenn:skipForwardNoBackwPass', ...
      '`skipForward` valid only when backward pass is computed.');
  end
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
  bnormCudnn = {'NoCuDNN'} ; % ours seems slighty faster
else
  cudnn = {'NoCuDNN'} ;
  bnormCudnn = {'NoCuDNN'} ;
end

switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  if opts.skipForward
    error('simplenn:skipForwardEmptyRes', ...
    'RES structure must be provided for `skipForward`.');
  end
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'stats', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end

if ~opts.skipForward
  res(1).x = x ;
end

% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'conv_mask'
      posTempX=repmat(l.posTemp{1},[1,1,1,size(res(i).x,4)]);
      posTempY=repmat(l.posTemp{2},[1,1,1,size(res(i).x,4)]);
      [h,w,depth,batchS]=size(res(i).x);
      [net.layers{i}.mu_x,net.layers{i}.mu_y,net.layers{i}.sqrtvar]=getMu(res(i).x);
      mask=getMask(net.layers{i},h,w,batchS,depth,posTempX,posTempY);
      input=res(i).x.*mask;
      clear mask posTempX posTempY
      res(i+1).x = vl_nnconv(max(input,0), l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        'dilate', l.dilate, ...
        l.opts{:}, ...
        cudnn{:}) ;
      clear input
    case 'convt'
      res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
        'crop', l.crop, ...
        'upsample', l.upsample, ...
        'numGroups', l.numGroups, ...
        l.opts{:}, ...
        cudnn{:}) ;

    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;

    case {'normalize', 'lrn'}
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;

    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;

    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case {'ourloss_logistic','ourloss'}
      res(i+1).x = our_vl_nnloss(res(i).x, l.class,'logistic');
    case 'ourloss_softmaxlog'
      [~,tmp]=max(l.class,[],3);
      res(i+1).x = vl_nnloss(res(i).x,gpuArray(tmp));
      clear tmp
    case 'deconvloss'
      [ih,iw,~,~]=size(res(i).x);
      tmp=res(1).x(round(linspace(1,size(res(1).x,1),ih)),round(linspace(1,size(res(1).x,2),iw)),:,:);
      res(i+1).x=mean(mean(mean(mean((res(i).x-tmp).^2,1),2),3),4);
      clear tmp
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;

    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;

    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;

    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;

    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;

    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end

    case 'bnorm'
      if testMode
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                                'moments', l.weights{3}, ...
                                'epsilon', l.epsilon, ...
                                bnormCudnn{:}) ;
      else
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                                'epsilon', l.epsilon, ...
                                bnormCudnn{:}) ;
      end

    case 'pdist'
      res(i+1).x = vl_nnpdist(res(i).x, l.class, l.p, ...
        'noRoot', l.noRoot, ...
        'epsilon', l.epsilon, ...
        'aggregate', l.aggregate, ...
        'instanceWeights', l.instanceWeights) ;

    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;

    otherwise
      error('Unknown layer type ''%s''.', l.type) ;
  end

  % optionally forget intermediate results
  needsBProp = doder && i >= backPropLim;
  forget = opts.conserveMemory && ~needsBProp ;
  if i > 1
    lp = net.layers{i-1} ;
    % forget RELU input, even for BPROP
    forget = forget && (~needsBProp || (strcmp(l.type, 'relu') && ~lp.precious)) ;
    forget = forget && ~(strcmp(lp.type, 'loss') || strcmp(lp.type, 'softmaxloss')) ;
    forget = forget && ~lp.precious ;
  end
  if forget
    res(i).x = [] ;
  end

  if gpuMode && opts.sync
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:backPropLim
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'pad', l.pad, ...
          'stride', l.stride, ...
          'dilate', l.dilate, ...
          l.opts{:}, ...
          cudnn{:}) ;
      case 'conv_mask'
          posTempX=repmat(l.posTemp{1},[1,1,1,size(res(i).x,4)]);
          posTempY=repmat(l.posTemp{2},[1,1,1,size(res(i).x,4)]);
          [h,w,depth,batchS]=size(res(i).x);
          mask=getMask(l,h,w,batchS,depth,posTempX,posTempY);
          input=res(i).x.*max(mask,0);
          clear posTempY mask_y
          [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
              vl_nnconv(input,l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
              'pad', l.pad, ...
              'stride', l.stride, ...
              'dilate', l.dilate, ...
              l.opts{:}, ...
              cudnn{:}) ;
          
          
          depthList=find(l.filter>0);
          labelNum=size(net.layers{end}.class,3);
          if(labelNum==1)
              theClass=net.layers{end}.class;
              Div=struct('depthList',depthList,'posList',find(theClass==1));
          else
              theClass=max(net.layers{end}.class,[],3);
              if(isempty(l.sliceMag))
                  Div=struct('depthList',depthList,'posList',find(theClass==1));
              else
                  sliceM=l.sliceMag;
                  Div=repmat(struct('depthList',[],'posList',[]),[1,labelNum]);
                  [~,idx]=max(sliceM(depthList,:),[],2);
                  for lab=1:labelNum
                      Div(lab).depthList=sort(depthList(idx==lab));
                      Div(lab).posList=find(net.layers{end}.class(:,:,lab,:)==1);
                  end
              end
          end
          
          
          imgNum=size(net.layers{end}.class,4);
          
          alpha=0.5;
          res(i).dzdx=res(i).dzdx.*max(mask,0);
          
          if(sum(l.filter==1)>0)
              l.strength=reshape(mean(mean(res(i).x.*mask,1),2),[depth,batchS]);
              alpha_logZ_pos=reshape(log(mean(exp(mean(mean(res(i).x.*mask(:,:,end:-1:1,end:-1:1),1),2)./alpha),4)).*alpha,[depth,1]);
              alpha_logZ_neg=reshape(log(mean(exp(mean(mean(-res(i).x,1),2)./alpha),4)).*alpha,[depth,1]);
              alpha_logZ_pos(isinf(alpha_logZ_pos))=max(alpha_logZ_pos(isinf(alpha_logZ_pos)==0));
              alpha_logZ_neg(isinf(alpha_logZ_neg))=max(alpha_logZ_neg(isinf(alpha_logZ_neg)==0));
          end
          
          for lab=1:numel(Div)
              if(numel(Div)==1)
                  w_pos=1;
                  w_neg=1;
              else
                  if(labelNum>10)
                      w_pos=0.5./(1/labelNum);
                      w_neg=0.5./(1-1/labelNum);
                  else
                      w_pos=0.5./net.layers{end}.density(lab);
                      w_neg=0.5./(1-net.layers{end}.density(lab));
                  end
              end
              
              %% For parts
              mag=ones(depth,imgNum)./(1/net.layers{end}.iter)./l.mag; %1.2;
              dList=Div(lab).depthList;
              dList=dList(l.filter(dList)==1);
              if(~isempty(dList))
                  list=Div(lab).posList;
                  if(~isempty(list))
                      strength=exp(l.strength(dList,list)./alpha).*(l.strength(dList,list)-repmat(alpha_logZ_pos(dList),[1,numel(list)])+alpha);
                      strength(isinf(strength))=max(strength(isinf(strength)==0));
                      strength(isnan(strength))=0;
                      strength=reshape(strength./(repmat(mean(strength,2),[1,numel(list)]).*mag(dList,list)),[1,1,numel(dList),numel(list)]);
                      strength(isnan(strength))=0;
                      strength(isinf(strength))=max(strength(isinf(strength)==0));
                      res(i).dzdx(:,:,dList,list)=res(i).dzdx(:,:,dList,list)-mask(:,:,dList,list).*repmat(strength,[h,w,1,1]).*(0.00001*w_pos);
                  end
                  
                  list_neg=setdiff(1:batchS,Div(lab).posList);
                  if(~isempty(list_neg))
                      strength=reshape(mean(mean(res(i).x(:,:,dList,list_neg),1),2),[numel(dList),numel(list_neg)]);
                      strength=exp(-strength./alpha).*(-strength-repmat(alpha_logZ_neg(dList),[1,numel(list_neg)])+alpha);
                      strength(isinf(strength))=max(strength(isinf(strength)==0));
                      strength(isnan(strength))=0;
                      strength=reshape(strength./(repmat(mean(strength,2),[1,numel(list_neg)]).*mag(dList,list_neg)),[1,1,numel(dList),numel(list_neg)]);
                      strength(isnan(strength))=0;
                      strength(isinf(strength))=max(strength(isinf(strength)==0));
                      res(i).dzdx(:,:,dList,list_neg)=res(i).dzdx(:,:,dList,list_neg)+repmat(reshape(strength,[1,1,numel(dList),numel(list_neg)]),[h,w,1,1]).*(0.00001*w_neg);
                  end
              end
          end
          
          
          beta=3;
          dzdw{3}=gpuArray(zeros(depth,1,'single'));
          for lab=1:numel(Div)
              dList=Div(lab).depthList;
              list=Div(lab).posList;
              tmp=sum(l.strength(dList,list).*l.sqrtvar(dList,list),2)./sum(l.strength(dList,list),2);
              tmp=max(min(beta./tmp,3.0),1.5);
              tmp=(tmp-l.weights{3}(dList)).*(-10000);
              dzdw{3}(dList)=gpuArray(single(tmp));
          end
          clear mask alpha_logZ_pos alpha_logZ_neg strength
      case 'convt'
        [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
          vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
          'crop', l.crop, ...
          'upsample', l.upsample, ...
          'numGroups', l.numGroups, ...
          l.opts{:}, ...
          cudnn{:}) ;

      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, ...
                                cudnn{:}) ;

      case {'normalize', 'lrn'}
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;

      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;

      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case {'ourloss_logistic','ourloss'}
        res(i).dzdx = our_vl_nnloss(res(i).x, l.class,'logistic',res(i+1).dzdx);
      case 'ourloss_softmaxlog'
        [~,tmp]=max(l.class,[],3);
        res(i).dzdx = vl_nnloss(res(i).x,gpuArray(tmp),res(i+1).dzdx);
        res(i).dzdx=res(i).dzdx.*(size(l.class,3)); %%%%%%%%%%%%%%%%%%%%%%%%%
        clear tmp
      case 'deconvloss'
        [ih,iw,~]=size(res(i).x);
        tmp=res(1).x(round(linspace(1,size(res(1).x,1),ih)),round(linspace(1,size(res(1).x,2),iw)),:,:);
        res(i).dzdx=(res(i).x-tmp)./numel(tmp);
        clear tmp
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;

      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end

      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;

      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;

      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;

      case 'dropout'
        if testMode
          res(i).dzdx = res(i+1).dzdx ;
        else
          res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end

      case 'bnorm'
        [res(i).dzdx, dzdw{1}, dzdw{2}, dzdw{3}] = ...
          vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, ...
                     'epsilon', l.epsilon, ...
                     bnormCudnn{:}) ;
        % multiply the moments update by the number of images in the batch
        % this is required to make the update additive for subbatches
        % and will eventually be normalized away
        dzdw{3} = dzdw{3} * size(res(i).x,4) ;

      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.class, ...
          l.p, res(i+1).dzdx, ...
          'noRoot', l.noRoot, ...
          'epsilon', l.epsilon, ...
          'aggregate', l.aggregate, ...
          'instanceWeights', l.instanceWeights) ;

      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end % layers

    switch l.type
      case {'conv', 'convt', 'bnorm','conv_mask','conv_aNet'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
        if ~isempty(opts.parameterServer) && ~opts.holdOn
          for j = 1:numel(res(i).dzdw)
            opts.parameterServer.push(sprintf('l%d_%d',i,j),res(i).dzdw{j}) ;
            res(i).dzdw{j} = [] ;
          end
        end
    end
    if opts.conserveMemory && ~net.layers{i}.precious && i ~= n
      res(i+1).dzdx = [] ;
      res(i+1).x = [] ;
    end
    if gpuMode && opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if i > 1 && i == backPropLim && opts.conserveMemory && ~net.layers{i}.precious
    res(i).dzdx = [] ;
    res(i).x = [] ;
  end
end
