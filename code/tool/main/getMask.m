function mask=getMask(l,h,w,batchS,depth,posTempX,posTempY)
mask=abs(posTempX-repmat(l.mu_x,[h,w,1]));
mask=mask+abs(posTempY-repmat(l.mu_y,[h,w,1]));
mask=max(1-mask.*repmat(reshape(l.weights{3},[1,1,depth]),[h,w,1,batchS]),-1);
mask(:,:,l.filter~=1,:)=1;
end
