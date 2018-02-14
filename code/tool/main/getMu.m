function [mu_x,mu_y,sqrtvar]=getMu(x)
IsUseMax=false;

[h,w,depth,batchS]=size(x);
x=reshape(x,[h*w,depth,batchS]);
if(IsUseMax)
    [maxX,p]=max(x,[],1);
    p=reshape(p,[1,1,depth,batchS]);
    mu_y=ceil(p./h);
    mu_x=p-(mu_y-1).*h;
    sqrtvar=[];
else
    tmp_x=gpuArray(single(repmat((1:h)',[w,depth,batchS])));
    tmp_y=gpuArray(single(reshape(repmat(1:w,[h,1,depth,batchS]),[h*w,depth,batchS])));
    sumX=max(sum(x,1),0.000000001);
    mu_x=max(round(sum(tmp_x.*x,1)./sumX),1);
    mu_y=max(round(sum(tmp_y.*x,1)./sumX),1);
    sqrtvar=sqrt((sum((tmp_x-repmat(mu_x,[h*w,1,1])).^2.*x,1)+sum((tmp_y-repmat(mu_y,[h*w,1,1])).^2.*x,1))./sumX);
    clear sumX tmp_x tmp_y
    [maxX,~]=max(x,[],1);
    p=reshape(mu_x+(mu_y-1).*h,[1,1,depth,batchS]);
end
tmp=linspace(-1,1,h);
mu_x=reshape(tmp(mu_x),[1,1,depth,batchS]);
mu_y=reshape(tmp(mu_y),[1,1,depth,batchS]);
sqrtvar=reshape(sqrtvar,[depth,batchS]);
end
