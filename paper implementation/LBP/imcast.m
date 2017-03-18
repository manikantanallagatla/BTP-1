function imout = imcast (img, outcls, varargin)

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (nargin == 3 && ! strcmpi (varargin{1}, "indexed"))
    error ("imcast: third argument must be the string \"indexed\"");
  endif

  incls = class (img);
  if (strcmp (outcls, incls))
    imout = img;
    return
  endif

  ## we are dealing with indexed images
  if (nargin == 3)
    if (! isind (img))
      error ("imcast: input should have been an indexed image but it is not.");
    endif

    ## Check that the new class is enough to hold all the previous indices
    ## If we are converting to floating point, then we don't bother
    ## check the range of indices. Also, note that indexed images of
    ## integer class are always unsigned.

    ## we will be converting a floating point image to integer class
    if (strcmp (outcls, "single") || strcmp (outcls, "double"))
      if (isinteger (img))
        imout = cast (img, outcls) +1;
      else
        imout = cast (img, outcls);
      endif

    ## we will be converting an indexed image to integer class
    else
      if (isinteger (img) && intmax (incls) > intmax (outcls) && max (img(:)) > intmax (outcls))
          error ("imcast: IMG has too many colours '%d' for the range of values in %s",
            max (img(:)), outcls);
      elseif (isfloat (img))
        imax = max (img(:)) -1;
        if (imax > intmax (outcls))
          error ("imcast: IMG has too many colours '%d' for the range of values in %s",
            imax, outcls);
        endif
        img -= 1;
      endif
      imout = cast (img, outcls);
    endif

  ## we are dealing with "normal" images
  else
    problem = false; # did we found a bad conversion?
    switch (incls)

      case {"double", "single"}
        switch (outcls)
          case "uint8",  imout = uint8  (img * 255);
          case "uint16", imout = uint16 (img * 65535);
          case "int16",  imout = int16 (double (img * uint16 (65535)) -32768);
          case {"double", "single"}, imout = cast (img, outcls);
          otherwise, problem = true;
        endswitch

      case {"uint8"}
        switch (outcls)
          case "double", imout = double (img) / 255;
          case "single", imout = single (img) / 255;
          case "uint16", imout = uint16 (img) * 257; # 257 comes from 65535/255
          case "int16",  imout = int16 ((double (img) * 257) -32768); # 257 comes from 65535/255
          otherwise, problem = true;
        endswitch

      case {"uint16"}
        switch (outcls)
          case "double", imout = double (img) / 65535;
          case "single", imout = single (img) / 65535;
          case "uint8",  imout = uint8 (img / 257); # 257 comes from 65535/255
          case "int16",  imout = int16 (double (img) -32768);
          otherwise, problem = true;
        endswitch

      case {"logical"}
        switch (outcls)
          case {"double", "single"}
            imout = cast (img, outcls);
          case {"uint8", "uint16", "int16"}
            imout = repmat (intmin (outcls), size (img));
            imout(img) = intmax (outcls);
          otherwise
            problem = true;
        endswitch

      case {"int16"}
        switch (outcls)
          case "double", imout = (double (img) + 32768) / 65535;
          case "single", imout = (single (img) + 32768) / 65535;
          case "uint8",  imout = uint8 ((double (img) + 32768) / 257); # 257 comes from 65535/255
          case "uint16", imout = uint16 (double (img) + 32768);
          otherwise, problem = true;
        endswitch

      otherwise
        error ("imcast: unknown image of class \"%s\"", incls);

    endswitch
    if (problem)
      error ("imcast: unsupported TYPE \"%s\"", outcls);
    endif
  endif

endfunction