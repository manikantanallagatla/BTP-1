function bool = isrgb (img)

  if (nargin != 1)
    print_usage;
  endif

  bool = false;
  if (isimage (img) && ndims (img) < 5 && size (img, 3) == 3)
    if (isfloat (img))
      bool = ispart (@is_float_image, img);
    elseif (any (isa (img, {"uint8", "uint16", "int16"})))
      bool = true;
    endif
  endif

endfunction