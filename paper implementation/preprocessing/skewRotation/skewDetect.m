function r=skewDetect(o_img)

%source :- unknown

global next_point;
global theta_max;           % angle in degrees 
       theta_max =60;
global T_ratio;             % ratio of Xwin for T 
       T_ratio=0.05;
global lamda lamda_max;    

Im = not(o_img);         % read Image
[Ywin, Xwin] = size(Im);
Im_RLSA = logical(zeros(Ywin, Xwin));           % zeros for Run Length Smoothing Algorithm
vertical_lines = logical(zeros(Ywin, Xwin));    % vertical lines linek(y)   p.1506 (1)

% subplot(2, 2, 1), imshow(not(Im));  title('first Text Image ');

T =  fix(T_ratio * Xwin);   % thresold T from p.1506
thresold = false;
 

%RLSA start
for i=1:Ywin
    j=1;
    while j<Xwin
        Im_RLSA(i,j)=Im(i,j);
        if ( Im(i,j)==1 ) && ( Im(i,j+1)==0)
            for k=j+2:min(j+1+T,Xwin)
                if (Im(i, k)==1) && (thresold==false)
                    thresold = true; 
                    next_point=k;
                    break;
                end
            end
            if (thresold == true)
                for k=j+1:next_point , Im_RLSA(i,k)=1; end
                j=next_point - 1;
                thresold = false;
            end
        end
        j=j+1;
    end
end
%RLSA end
% subplot(2, 2, 2), imshow(not(Im_RLSA)); title('after RLSA');



T_2 = fix(T/2);
x_win_3 = fix( Xwin / 3);
D1=x_win_3; D2=2*x_win_3;

%for j=1+T_2:Xwin-T_2
% vertical lines p.1509 Fig 3
for j=D1:D1:D2+1
    for i=1:Ywin
        %vertical_lines(i,j) = not(isempty( find(Im_RLSA(i,j-T_2:j+T_2)>1, 1)));
        for k=j-T_2:j+T_2
            if Im_RLSA(i,k)>0               % p1506 [1]
                vertical_lines(i,j) = 1;    %
                break                       % for faster exit from loop
            end                             %
        end
        %if sum(Im_RLSA(i,j-T_2:j+T_2)) >0 , vertical_lines(i,j) = 1; end
    end
end



L = fix( (D2-D1) * tan(2*pi*theta_max / 360.0) );   % how many pixels for max  p.1507  [2]

P = (xcorr(double( vertical_lines(:,D1)) , double( vertical_lines(:,D2)) , L )); % P() 1508 Fig 4 for =1...2*L+1
lamda_max = find (P == max(P)) - (L + 1);                   % find the position of max P()
% sprintf('max = %f', lamda_max)
skew_theta = atan(lamda_max / (D2 - D1)) * 360 / (2 * pi);  % the  from p1508 [5] and convert in degrees by 360/(2 pi)

r=skew_theta;
