function retval = isimage (img)
  retval = ((isnumeric (img) || islogical (img)) && ! issparse (img)
            && ! isempty (img) && isreal (img));
endfunction