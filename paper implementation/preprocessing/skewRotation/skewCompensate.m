file_input = 'result1.jpg';
file_output = 'result2.jpg';
img = imread(file_input);
%%%detect skew angle
angle = skewDetect(img);

%%%code for rotating
% img = imrotate(img, -1*angle,'bilinear','crop');

s   = ceil(size(img)/2);
imP = padarray(img, s(1:2), 'replicate', 'both');
imR = imrotate(imP, -1*angle);
S   = ceil(size(imR)/2);
imF = imR(S(1)-s(1):S(1)+s(1)-1, S(2)-s(2):S(2)+s(2)-1, :); %// Final form


imwrite(imF, file_output);