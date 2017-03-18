function BW = im2bw (img, cmap, thresh = 0.5)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  elseif (nargin == 3 && ! isind (img))
    error ("im2bw: IMG must be an indexed image when are 3 input arguments");
  elseif (nargin == 3 && ! iscolormap (cmap))
    error ("im2bw: CMAP must be a colormap");
  elseif (nargin == 2)
    thresh = cmap;
  endif

  if (! isimage (img))
    error ("im2bw: IMG must be an image");
  elseif (! ischar (thresh) && ! (isnumeric (thresh) && isscalar (thresh)
                                  && thresh >= 0 && thresh <= 1))
    error ("im2bw: THRESHOLD must be a string or a scalar in the interval [0 1]");
  endif

  if (islogical (img))
    warning ("im2bw: IMG is already binary so nothing is done");
    tmp = img;

  else
    ## Convert img to gray scale
    if (nargin == 3)
      ## indexed image (we already checked that is indeed indexed earlier)
      img = ind2gray (img, cmap);
    elseif (isrgb (img))
      img = rgb2gray (img);
    else
      ## Everything else, we do nothing, no matter how many dimensions
    endif

    if (ischar (thresh))
      thresh = graythresh (img(:), thresh);
    endif

    ## Convert the threshold value to same class as the image which
    ## is faster and saves more memory than the opposite.
    if (isinteger (img))
      ## We do the conversion from double to int ourselves (instead
      ## of using im2uint* functions), because those functions round
      ## during the conversion but we need thresh to be the limit.
      ## See bug #46390.
      cls = class(img);
      I_min = double (intmin (cls));
      I_range = double (intmax (cls)) - I_min;
      thresh = cast (floor ((thresh * I_range) + I_min), cls);
    elseif (isfloat (img))
      ## do nothing
    else
      ## we should have never got here in the first place anyway
      error ("im2bw: unsupported image of class '%s'", class (img));
    endif

    tmp = (img > thresh);
  endif

  if (nargout > 0)
    BW = tmp;
  else
    imshow (tmp);
  endif

endfunction