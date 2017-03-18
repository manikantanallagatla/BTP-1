function [N] = templateMatching(template_img, background_img,background_img1)

template = (imread(template_img));
background = (imread(background_img));

background1 = (imread(background_img1));
if size(template,3)==3 
    template=rgb2gray(template);
%     figure,imshow(im);
end
if size(background,3)==3 
    background=rgb2gray(background);
%     figure,imshow(im);
end

% threshold = graythresh(template);
% template =~im2bw(template,threshold);
% background = graythresh(background);
% background =~im2bw(background,threshold);

%% calculate padding
bx = size(background, 2); 
by = size(background, 1);
tx = size(template, 2); % used for bbox placement
ty = size(template, 1);
tx=tx+4;
ty=ty+4;

%% fft
%c = real(ifft2(fft2(background) .* fft2(template, by, bx)));

%// Change - Compute the cross power spectrum
Ga = fft2(background);
Gb = fft2(template, by, bx);
c = real(ifft2((Ga.*conj(Gb))./abs(Ga.*conj(Gb))));

%% find peak correlation
[max_c, imax]   = max(abs(c(:)));
[ypeak, xpeak] = find(c == max(c(:)));
%figure; surf(c), shading flat; % plot correlation    

%% display best match
hFig = figure;
hAx  = axes;

%// New - no need to offset the coordinates anymore
%// xpeak and ypeak are already the top left corner of the matched window
position = [xpeak(1), ypeak(1), tx, ty];
imshow(background1, 'Parent', hAx);
imrect(hAx, position);

% maxcorr = max(c(:));
% thresh = 0.85*maxcorr;
% %0.85 is optimum
% b = c>thresh;
% N = sum(sum(b));
% imshow(background, 'Parent', hAx);
% [ b, ix ] = sort( c(:), 'descend' );
% [ rr, cc ] = ind2sub( size(c), ix(1:N) );
% for ii = 1 : N
%     disp( M( rr(ii), cc(ii) ) )
%     [ypeak, xpeak] = find(c == c(rr(ii), cc(ii)));
%     position = [xpeak(1), ypeak(1), tx, ty];
%    imrect(hAx, position);
% end