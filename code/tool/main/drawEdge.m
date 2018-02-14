function I=drawEdge(wei,Iori)
Itmp=edge(wei,'canny');
Itmp=single(imfilter(Itmp,ones(5))>0).*255;
I=zeros(size(Iori),'uint8');
Iori=single(Iori);
I(:,:,1)=max(Iori(:,:,1).*((wei+1)./2),Itmp);
I(:,:,2)=min(Iori(:,:,2).*((wei+1)./2),255-Itmp);
I(:,:,3)=min(Iori(:,:,3).*((wei+1)./2),255-Itmp);
end
