function bw = bwareaopen (bw, lim, conn)

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (! ismatrix (bw) || ! (isnumeric (bw) || islogical (bw)))
    error ("bwareaopen: BW must be an a numeric matrix");
  elseif (! isnumeric (lim) || ! isscalar (lim) || lim < 0 || fix (lim) != lim)
    error ("bwareaopen: LIM must be a non-negative scalar integer")
  endif

  if (nargin < 3)
    ## Defining default connectivity here because it's dependent
    ## on the first argument
    conn = conndef (ndims (bw), "maximal");
  else
    conn = make_conn ("bwareaopen", 3, ndims (bw), conn);
  endif

  ## Output is always of class logical
  bw = logical (bw);

  ## We only have work to do when lim > 1
  if (lim > 1)
    idx_list = bwconncomp (bw, conn).PixelIdxList;
    ind = cell2mat (idx_list (cellfun ("numel", idx_list) < lim)');
    bw(ind) = false;
  endif

endfunction
