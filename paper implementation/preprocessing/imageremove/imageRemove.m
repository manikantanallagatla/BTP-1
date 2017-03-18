close all;
im=imread('result2.jpg');
figure;
imshow(im);
if size(im,3)==3 
    im=rgb2gray(im);
%     figure,imshow(im);
end

threshold = graythresh(im);
i =~im2bw(im,1*threshold);
% figure,imshow(i);

% i = bwareaopen(i,30);
k=i;
[L Ne] = bwlabel(i);
% figure,imshow(Ne);
% figure,imshow(L);
countArray = zeros(1,Ne);
siz = size(i);
h = siz(1);
w = siz(2);
for x = 1:h
    for y = 1:w
        if(L(x,y)>0)
            countArray(L(x,y)) = 1+countArray(L(x,y));
        end
    end
end
for n=1:Ne
    countTemp = countArray(n);
    if(countTemp > 500)
        [f co]=find(L==n);
        for tempx = min(f)-1:max(f)+1
            for tempy = min(co)-1:max(co)+1
                im(tempx,tempy) = 255;
            end
        end
    end
end
figure;
imshow(im);
imwrite(im,'result3.jpg');