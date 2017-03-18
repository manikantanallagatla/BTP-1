function conn = conndef (num_dims, conntype)

  if (nargin != 2)
    print_usage ();
  elseif (! isnumeric (num_dims) || ! isscalar (num_dims) || num_dims <= 0 ||
          fix (num_dims) != num_dims)
    error ("conndef: NUM_DIMS must be a positive integer");
  elseif (! ischar (conntype))
    error ("conndef: CONNTYPE must be a string with type of connectivity")
  endif

  if (strcmpi (conntype, "minimal"))
    if (num_dims == 1)
      ## This case does not exist in Matlab
      conn = [1; 1; 1];
    elseif (num_dims == 2)
      ## The general case with the for loop below would also work for
      ## 2D but it's such a simple case we have it like this, no need
      ## to make it hard to read.
      conn = [0 1 0
              1 1 1
              0 1 0];
    else
      conn   = zeros (repmat (3, 1, num_dims));
      template_idx = repmat ({2}, [num_dims 1]);
      for dim = 1:num_dims
        idx = template_idx;
        idx(dim) = ":";
        conn(idx{:}) = 1;
      endfor
    endif

  elseif (strcmpi (conntype, "maximal"))
    if (num_dims == 1)
      ## This case does not exist in Matlab
      conn = [1; 1; 1];
    else
      conn = ones (repmat (3, 1, num_dims));
    endif

  else
    error ("conndef: invalid type of connectivity '%s'.", conntype);
  endif

endfunction