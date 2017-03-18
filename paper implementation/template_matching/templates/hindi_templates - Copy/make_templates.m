close all;
im=imread('set7.JPG');
figure;
imshow(im);
if size(im,3)==3 
    im=rgb2gray(im);
%     figure,imshow(im);
end

threshold = graythresh(im);
i =~im2bw(im,1*threshold);
% figure,imshow(i);

i = bwareaopen(i,30);
k=i;
[L Ne]= bwlabel(i);

index = 644;

for n=1:Ne
    [f co]=find(L==n);
    inew=im(min(f):max(f),min(co):max(co));
    name =  strcat(int2str(index),'.jpg');
    index = index+1;
    imwrite(inew,name);
end
figure;
imshow(im);