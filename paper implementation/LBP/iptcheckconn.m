function iptcheckconn (con, func_name, var_name, pos)

  ## thanks to Oldak in ##matlab for checking the validity of connectivities
  ## with more than 2D and the error messages

  if (nargin != 4)
    print_usage;
  elseif (!ischar (func_name))
    error ("Argument func_name must be a string");
  elseif (!ischar (var_name))
    error ("Argument var_name must be a string");
  elseif (!isnumeric (pos) || !isscalar (pos) || !isreal (pos) || pos <= 0 || rem (pos, 1) != 0)
    error ("Argument pos must be a real positive integer");
  endif

  base_msg = sprintf ("Function %s expected input number %d, %s, to be a valid connectivity specifier.\n       ", ...
                      func_name, pos, var_name);

  ## error ends in \n so the back trace of the error is not show. This is on
  ## purpose since the whole idea of this function is already to give a properly
  ## formatted error message
  if (!any (strcmp (class (con), {'logical', 'double'})) || !isreal (con) || !isnumeric (con))
    error ("%sConnectivity must be a real number of the logical or double class.\n", base_msg);
  elseif (isscalar (con))
    if (!any (con == [1 4 6 8 18 26]))
      error ("%sIf connectivity is a scalar, must belong to the set [1 4 6 8 18 26].\n", base_msg);
    endif
  elseif (ismatrix (con))
    center_index = ceil(numel(con)/2);
    if (any (size (con) != 3))
      error ("%sIf connectivity is a matrix, all dimensions must have size 3.\n", base_msg);
    elseif (!all (con(:) == 1 | con(:) == 0))
      error ("%sIf connectivity is a matrix, only 0 and 1 are valid.\n", base_msg);
    elseif (con(center_index) != 1)
      error ("%sIf connectivity is a matrix, central element must be 1.\n", base_msg);
    elseif (!all (con(1:center_index-1) == con(end:-1:center_index+1)))
      error ("%sIf connectivity is a matrix, it must be symmetric relative to its center.\n", base_msg);
    endif
  else
    error ("%s\n", base_msg);
  endif

endfunction